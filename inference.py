import sys
import argparse
import random
import math
import logging
import os
import gc
import copy
from tqdm import tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from skimage.transform import resize
from peft import LoraConfig, get_peft_model
import torch.utils.data as tud
from aurora.model.aurora_vq import AuroraSmallWithVQ, AuroraWithVQ, AuroraHighResWithVQ
from aurora import rollout
from aurora.batch import Batch, Metadata
from aurora.utils.metrics import rmse, mae
from dataset import CWA_ignore_missing
from torch.utils.checkpoint import checkpoint
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Force reset CUDA environment to avoid conflicts
os.environ.pop("CUDA_DEVICE_ORDER", None)
os.environ.pop("CUDA_CACHE_DISABLE", None)
os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

# Set clean CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Force CUDA initialization before any torch operations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Enable TF32 to reduce memory/compute pressure on supported GPUs
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

def kelvin_to_celsius(x): return x - 273.15
def geopotential_to_height(x): return x / 9.80665

class LogFlagFilter(logging.Filter):
    def __init__(self, log_flag):
        super().__init__()
        self.log_flag = log_flag

    def filter(self, record):
        return self.log_flag


class Aurora_trainer:
    def __init__(self, args=None):
        self.args = args
        self.valid_count_per_batch = []
        self.root_dir = "./"
        self.tranditional_model_leadtime = 4
        self.checkpoint_path = self.args.checkpoint_path

        # Prefer checkpoint filename for inference prefix; fallback to codebook
        if self.checkpoint_path:
            ckpt_name = os.path.basename(self.checkpoint_path)
            stem = ckpt_name[:-4] if ckpt_name.endswith('.pth') else os.path.splitext(ckpt_name)[0]
            self.file_prefix = f"test_{stem}"
        elif self.args.dictionary_path:
            model_name = self.args.dictionary_path.split("/")[-1][:-4]
            self.file_prefix = f"test_{model_name}"
        else:
            self.file_prefix = "test_default"

        self.log_dir = f"{self.root_dir}/logs" ### need model name or it will conflate
        self.saved_model_dir = f"{self.root_dir}/weights/" ### need model name or it will conflate
        self.metric_dir = f"{self.root_dir}/metrics/" ### need model name or it will conflate
        self.test_on_map_dir = f"{self.root_dir}/plot/"

        self.log_file_name = os.path.join(self.log_dir, f"{args.predict_time}.log")
        self.metric_file_name_rmse = os.path.join(self.metric_dir, f"{args.predict_time}_RMSE.csv")
        self._create_dirs()

        self.logging_filter = LogFlagFilter(True)
        self.logger = self._setup_logger()
        self.logger.setLevel(logging.INFO)
        self.logger.addFilter(LogFlagFilter(True))

        # hyperparameters
        self.leadtime = 6
        self.rollout_step = 1
        self.batch_size = 3
        self.global_step = 0
        self.epoch = 0

        # codebook settings
        self.codebook_size = 4096
        self.dictionary_path = args.dictionary_path

        # Aurora Model setting
        self.vars = {
            "surf_vars": ["2t", "10u", "10v", "msl"],
            "atmos_vars": ["t", "u", "v", "q", "z"],
            "static_vars": ["z", "slt", "lsm"],
        }

        # Dataset
        self.num_workers = 12
        self.idx2real_pressurelevel = [
            1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
        ]

        # VQVAE settings : will not alter
        self.vq_sigma = 0.25
        self.softvq_tau = 0.25
        self.softvq_entropy_ratio = 0.15
        self.softvq_show_usage = True
        self.softvq_l2_norm = True
        self.softvq_chunk_size_tokens = 1024
        self.use_vq_dconv = True
        self.vq_dconv_atoms = 4096

        # Record
        self.atmos_levels = []
        self.mean_metric_per_step_rmse = [{} for _ in range(self.rollout_step)]
        self.scaler = GradScaler()

    @staticmethod
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    def _set_seed(self, seed=42):
        sys.setrecursionlimit(100000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_dirs(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        if not os.path.exists(self.test_on_map_dir):
            os.makedirs(self.test_on_map_dir, exist_ok=True)

        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir, exist_ok=True)


    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )
        return logging.getLogger()

    def _find_codebook_tensor(self, checkpoint):
        """recursively find codebook tensor in a checkpoint dict and return codebook tensor or None

        Strategy:
          1) find codebook tensor alias keys (ex : candidate_keys).
          2) DFS each ckpt keys to find those who have ['embedding', 'codebook', 'aurora_vq'].
          3) Shape-based fallback: prefer tensors whose shape matches
             [n_e, e_dim], [e_dim, n_e], or [1, n_e, e_dim], where n_e == codebook_size.
          4) DFS each ckpt keys do Shape-based fallback
        """
        if not isinstance(checkpoint, dict):
            return None

        # (1) direct/common keys
        candidate_keys = [
            'aurora_vq.embedding.weight',
            'model.aurora_vq.embedding.weight',
            'aurora_vq.softvq.embedding',
            'aurora_vq.softvq.embedding.weight',
            'model.aurora_vq.softvq.embedding',
            'model.aurora_vq.softvq.embedding.weight',
            'embedding.weight',
            'embedding',
            'codebook',
            'model.codebook',
        ]
        for k in candidate_keys:
            if k in checkpoint and isinstance(checkpoint[k], torch.Tensor):
                return checkpoint[k]

        # unwrap common wrappers to deeper checkpoint
        wrapped = None
        for wrap_key in ['model_state_dict', 'state_dict']:
            if wrap_key in checkpoint and isinstance(checkpoint[wrap_key], dict):
                wrapped = checkpoint[wrap_key]
                break
        search_space = wrapped if wrapped is not None else checkpoint

        # (2) yield the dict object by dfs. ex : (a, a.l1_0, a.l1_0.l2_0, a.l1_1)
        def iter_tensors(d, prefix=''):
            for key, val in d.items():
                full = f"{prefix}{key}"
                if isinstance(val, dict):
                    yield from iter_tensors(val, full + '.')
                elif isinstance(val, torch.Tensor):
                    yield full, val

        name_hint_candidates = []
        for full_k, t in iter_tensors(search_space):
            low = full_k.lower()
            if ('embedding' in low) or ('codebook' in low) or ('aurora_vq' in low):
                name_hint_candidates.append((full_k, t))

        # (3) shape-based selection
        def pick_best_by_shape(cands):
            best = None
            best_score = -1
            for _, ten in cands:
                if not isinstance(ten, torch.Tensor):
                    continue
                shape = tuple(ten.shape)
                score = 0
                # Rank by how well the shape matches expected patterns
                if len(shape) == 2:
                    if shape[0] == self.codebook_size:
                        score = 3  # [n_e, e_dim]
                    elif shape[1] == self.codebook_size:
                        score = 2  # [e_dim, n_e] (needs transpose)
                elif len(shape) == 3:
                    # [num_codebooks, n_e, e_dim]
                    if shape[0] in (1,):
                        if shape[1] == self.codebook_size:
                            score = 4  # ideal
                        else:
                            score = 1
                if score > best_score:
                    best = ten
                    best_score = score
            return best

        if name_hint_candidates:
            best = pick_best_by_shape(name_hint_candidates)
            if best is not None:
                return best

        # final fallback: consider all tensors
        all_tensors = list(iter_tensors(search_space))
        if all_tensors:
            best = pick_best_by_shape(all_tensors)
            if best is not None:
                return best

        return None

    # initial model without ckpt
    def make_model(self, load_optimizer_state=False):
        model = AuroraSmallWithVQ(
            codebook_size=self.codebook_size,
            vq_sigma=self.vq_sigma,
            use_vq_dconv=True, # stage 1 stage 2 merged
            vq_dconv_atoms=self.codebook_size,
            lambda_rd=0.8,
            enable_lambda_generator = False,
            use_lora=False,
            autocast=False,
            freeze_stage2=False,
            input_args = self.args
        ).to('cpu')

        self.official_checkpoint_name = "aurora-0.25-small-pretrained.ckpt"
        missing_key, unexpected_key = model.load_checkpoint("microsoft/aurora", self.official_checkpoint_name, strict=False)
        missing_key = [key for key in missing_key if ("mr_proj_layers" not in key) and ("vq" not in key)]
        self.logger.info(f"official missing key = {missing_key}")
        self.logger.info(f"official unexpected key = {unexpected_key}")

        lambda_rd_tensor = model.lambda_rd
        if isinstance(lambda_rd_tensor, torch.Tensor):
            lambda_rd_value = torch.sigmoid(lambda_rd_tensor.detach().reshape(())).item()

        model = self.packaged_by_lora(model)
        model.configure_activation_checkpointing()
        return model

    def packaged_by_lora(self, model):
        # set the peft_config based on method selection
        encoder_lora = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.1,
            inference_mode=False,
            target_modules=["qkv", "proj"],
            layers_pattern="encoder_layers",
        )

        decoder_lora = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.1,
            inference_mode=False,
            target_modules=["qkv", "proj", "lin1", "lin2"],
            layers_pattern="decoder_layers",
        )
        for i, layer in enumerate(model.backbone.encoder_layers):
            model.backbone.encoder_layers[i] = get_peft_model(layer, encoder_lora)
        for i, layer in enumerate(model.backbone.decoder_layers):
            model.backbone.decoder_layers[i] = get_peft_model(layer, decoder_lora)

        return model

    def _load_codebook(self):

        try:
            self.logger.info(f"Loading Dictionary")

            file_ext = os.path.splitext(self.dictionary_path)[1].lower()
            codebook_tensor = None

            """
                Check if file is .npy (k-means codebook) or .pth (checkpoint)
                and get kmeans_array and codebook_tensor (trained codebook)
            """

            if file_ext == '.npy':
                # Direct .npy file (k-means codebook from k_means.py)
                kmeans_array = np.load(self.dictionary_path) # codebook_size, dim
                # Convert numpy array to torch tensor
                codebook_tensor = torch.from_numpy(kmeans_array).float()

            else:
                # .pth checkpoint file (trained codebook from BackboneSoftVQ_trainer)
                checkpoint = torch.load(self.dictionary_path, map_location='cpu', weights_only=False)

                # Find codebook tensor with robust search
                if isinstance(checkpoint, dict):
                    for container in [checkpoint.get('model_state_dict', {}), checkpoint.get('state_dict', {}), checkpoint]:
                        if not isinstance(container, dict):
                            continue

                        # Check for SoftVQ embedding (from BackboneSoftVQ_trainer)
                        if 'aurora_vq.softvq.embedding' in container and isinstance(container['aurora_vq.softvq.embedding'], torch.Tensor):
                            codebook_tensor = container['aurora_vq.softvq.embedding']
                            break

                        # Check for standard VQ embedding
                        if 'aurora_vq.embedding.weight' in container and isinstance(container['aurora_vq.embedding.weight'], torch.Tensor):
                            codebook_tensor = container['aurora_vq.embedding.weight']
                            break

                if codebook_tensor is None:
                    codebook_tensor = self._find_codebook_tensor(checkpoint)

            """
                load codebook_tensor into codebook
            """

            if codebook_tensor is not None and hasattr(self.model, 'vq_dconv'):

                # reshape to [n_e, e_dim] format
                emb_2d = codebook_tensor
                shape = tuple(emb_2d.shape)

                if len(shape) == 3 and shape[0] == 1:
                    # SoftVQ format: [1, n_e, e_dim] -> [n_e, e_dim]
                    emb_2d = emb_2d.reshape(shape[1], shape[2])
                elif len(shape) == 2:
                    pass
                else:
                    raise ValueError(f"Unsupported codebook tensor shape: {shape}")

                if emb_2d.shape[0] != self.codebook_size:
                    raise ValueError(
                        f"Codebook size mismatch: expected {self.codebook_size}, got {emb_2d.shape[0]}"
                    )

                # embedding dimension
                expected_e_dim = 256  # VQDConv channel_compression output dim
                actual_e_dim = emb_2d.shape[1] # n, dim

                if actual_e_dim == expected_e_dim:
                    # load into VQDConv D layer (codebook)
                    self.model.vq_dconv.load_codebook_weights(emb_2d)

            else:
                self.logger.warning("Could not find codebook tensor in checkpoint or VQDConv not found")

        except Exception as e:
            self.logger.error(f"Failed to load first stage codebook: {e}")
            raise


    def load_pass_resume_ckpt(self, checkpoint_path):
        stage1_codebook = None
        if self.dictionary_path and hasattr(self.model, 'vq_dconv') and hasattr(self.model.vq_dconv, 'D'):
            stage1_codebook = self.model.vq_dconv.D.weight.clone().detach()

        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']

        filtered_state = {}
        skipped = []

        for k, v in checkpoint.items():
            filtered_state[k] = v

        missing_key, unexpected_key = self.model.load_state_dict(filtered_state, strict=False)

        # Restore first stage codebook to ensure correct codebook is used for inference
        if stage1_codebook is not None and hasattr(self.model, 'vq_dconv') and hasattr(self.model.vq_dconv, 'D'):
            with torch.no_grad():
                self.model.vq_dconv.D.weight.copy_(stage1_codebook)


    def _init_model_for_testing(self):
        """Initialize model for testing: load official weights, merge LoRA, load codebook, then load Stage 2 checkpoint."""

        self.model = self.make_model()
        self.merge_unload_lora(self.model)

        if self.dictionary_path is not None:
            self._load_codebook()

        if self.checkpoint_path:
            self.load_pass_resume_ckpt(self.checkpoint_path)
        else:
            self.logger.warning("No checkpoint_path provided; using current model weights")

        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            self.logger.info(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("CUDA not available, using CPU")

    def merge_unload_lora(self, this_model):
        self.logger.info("Lora Merge and Unload")
        for i, layer in enumerate(this_model.backbone.encoder_layers):
            this_model.backbone.encoder_layers[i].merge_and_unload()

        for i, layer in enumerate(this_model.backbone.decoder_layers):
            this_model.backbone.decoder_layers[i].merge_and_unload()

    def metric_rmse_gen_csv(self):
        total_sample = len(self.test_dataset)
        df = pd.DataFrame()
        for step in range(self.rollout_step):
            metric_step={}
            for key in self.vars['surf_vars']:
                metric_step[key]=self.mean_metric_per_step_rmse[step][f"my_rmse"]['surf_vars'][key].item()/total_sample
            for key in self.vars['atmos_vars']:
                for idx, level in enumerate(self.atmos_levels):
                    metric_step[f"{key}_{level}"]=self.mean_metric_per_step_rmse[step][f"my_rmse"]['atmos_vars'][key][idx].item()/total_sample

            self.logger.info(f"Rollout {step} RMSE t2m =  {self.mean_metric_per_step_rmse[step][f'my_rmse']['surf_vars']['2t'].item()/total_sample}")
            self.logger.info(f"Rollout {step} RMSE u10 =  {self.mean_metric_per_step_rmse[step][f'my_rmse']['surf_vars']['10u'].item()/total_sample}")
            self.logger.info(f"Rollout {step} RMSE t850 =  {self.mean_metric_per_step_rmse[step][f'my_rmse']['atmos_vars']['t'][2].item()/total_sample}")
            self.logger.info(f"Rollout {step} RMSE z500 =  {self.mean_metric_per_step_rmse[step][f'my_rmse']['atmos_vars']['z'][5].item()/total_sample}")

            df[step+1]=metric_step
        df.to_csv(self.metric_file_name_rmse)

    def main_process(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

        gc.collect()
        self._set_seed()

        # Select Device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.logger.info(f"Using GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} : {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        else:
            self.device = torch.device('cpu')
            self.logger.warning("CUDA not available, using CPU")

        cpu_cnt = os.cpu_count() or 10
        optim_workers = max(0, min(self.num_workers, max(1, cpu_cnt - 2)))
        use_persistent = bool(optim_workers > 0)
        prefetch = 2 if optim_workers > 0 else None

        # testing path: init model exactly like train (merge old lora -> attach fresh lora -> load final)
        self._init_model_for_testing()

        # build test loader
        self.test_dataset  = CWA_ignore_missing(data_path=self.args.data_dir, leadtime=self.leadtime, step=self.rollout_step, dataset_time_range=args.predict_time, dev=False, dataset_time_type="use_hour", patch_size=4, whether_regrid=False, divergence_mode=None)

        # default worker by fork
        self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=optim_workers,
                persistent_workers=use_persistent,
                prefetch_factor=prefetch,
        )

        self.test_epoch()

    def prepare_aurora_data_pair(self, batch_input, batch_label):
        current_times = []
        predict_times = []
        for current_time in batch_input["time"]:
            current_times.append(datetime.strptime(current_time, '%Y%m%d%H'))
        for predict_time in batch_label[0]["time"]:
            predict_times.append(datetime.strptime(predict_time, '%Y%m%d%H'))

        static_vars = {k: batch_input["static_vars"][k][0] for k in self.vars["static_vars"]}
        lat = batch_input["lat"][0]
        lon = batch_input["lon"][0]
        self.atmos_levels = tuple(int(level) for level in batch_input["atmos_levels"][0])

        input = Batch(
            surf_vars=batch_input["surf_vars"],
            static_vars=static_vars,
            atmos_vars=batch_input["atmos_vars"],
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=current_times,
                atmos_levels=self.atmos_levels,
            )
        ).to(self.device)

        label = Batch(
            surf_vars=batch_label[0]["surf_vars"],
            static_vars=static_vars,
            atmos_vars=batch_label[0]["atmos_vars"],
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=predict_times,
                atmos_levels=self.atmos_levels,
            )
        ).to(self.device)

        with torch.no_grad():
            label_norm = label.normalise(surf_stats=self.model.surf_stats)

        return input, label, label_norm

    def make_loss_dict(self, pred, pred_low, label, label_norm, vq_loss, entropy_loss, batch_idx) :

        total_loss = 0
        loss_low_1_2 = 0.0
        loss_low_1_4 = 0.0
        div_loss = 0.0
        mae_loss = 0.0

        pred = pred.to(self.device)
        label_norm = label_norm.to(self.device)
        mae_loss = self.criterion(pred, label_norm)
        total_loss += mae_loss

        entropy_loss =  entropy_loss.to(self.device)
        entropy_loss = 0.0001 * entropy_loss
        total_loss = total_loss + entropy_loss

        loss_dict = {
            'reco_loss': mae_loss,
            'vq_loss': vq_loss,
            'total_loss' : total_loss,
            'loss_low_1_2': loss_low_1_2,
            'loss_low_1_4': loss_low_1_4,
            "loss_div": div_loss,
            "entropy_loss": entropy_loss,
        }

        return loss_dict

    def test_epoch(self):
        if not hasattr(self, 'model') or self.model is None:
            self.logger.error("Model not initialized for testing")
            return

        self.merge_unload_lora(self.model)

        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.inference_mode():
            for batch_idx, (test_input, test_labels) in enumerate(tqdm(self.test_loader, desc="Predicting", leave=False, ncols=100)):
                current_times = []
                predict_times = []

                for current_time in test_input["time"]:
                    current_times.append(datetime.strptime(current_time, '%Y%m%d%H'))
                for predict_time in test_labels[0]["time"]:
                    predict_times.append(datetime.strptime(predict_time, '%Y%m%d%H'))

                static_vars = {k: test_input["static_vars"][k][0] for k in self.vars["static_vars"]}
                lat = test_input["lat"][0]
                lon = test_input["lon"][0]
                self.atmos_levels = tuple(int(level) for level in test_input["atmos_levels"][0])
                B,T,H,W=test_input["surf_vars"][self.vars['surf_vars'][0]].shape

                input = Batch(
                    surf_vars=test_input["surf_vars"],
                    static_vars=static_vars,
                    atmos_vars=test_input["atmos_vars"],
                    metadata=Metadata(
                        lat=lat,
                        lon=lon,
                        time=current_times,
                        atmos_levels=self.atmos_levels,
                    )
                ).to(self.device)
                labels = [Batch(
                    surf_vars=test_labels[step]["surf_vars"],
                    static_vars=static_vars,
                    atmos_vars=test_labels[step]["atmos_vars"],
                    metadata=Metadata(
                        lat=lat,
                        lon=lon,
                        time=predict_times,
                        atmos_levels=self.atmos_levels,
                    )
                ).to(self.device) for step in range(self.rollout_step)]

                # rollout deal with VQ and non-VQ output
                if torch.cuda.is_available():
                    with autocast():
                        preds = [pred for pred in rollout(self.model, input, steps=self.rollout_step, leadtime=self.leadtime, batch_idx=batch_idx)]
                else:
                    preds = [pred for pred in rollout(self.model, input, steps=self.rollout_step, leadtime=self.leadtime, batch_idx=batch_idx)]

                for step in range(self.rollout_step):
                    pred = preds[step][0]
                    pred = pred.unnormalise(surf_stats=self.model.surf_stats)
                    # do unnormalise in rollout.py
                    label = labels[step]
                    self.valid_count_per_batch.append(preds[step][2])

                    # choose 2022010100, it will plot 2022010100 + step * leadtime
                    # ex: leadtime = 6, step = 1, it will plot 2022010112

                    lat_plot = lat.numpy()
                    lon_plot = lon.numpy()
                    # Match the variable naming / title style used in
                    # Taiwan-Aurora-Foundation-Model/plot/plot_hrrr_zero_shot.py
                    draw_vars_setting = {
                        "2t": ("surf_vars", kelvin_to_celsius, None, "Temperature at 2m (°C)"),
                        "10u": ("surf_vars", None, None, "U-wind at 10m (m/s)"),
                        "t850": ("atmos_vars", kelvin_to_celsius, 850, "Temperature at 850 hPa (°C)"),
                        "z500": ("atmos_vars", geopotential_to_height, 500, "Geopotential Height at 500 hPa (m)"),
                    }

                    for vname, (vtype, convert_method, target_level, plot_title) in draw_vars_setting.items():

                        if vtype == "surf_vars":
                            pred_v = pred.surf_vars[vname][0, 0].cpu().detach().numpy()
                            gt_v   = label.surf_vars[vname][0, 0].cpu().detach().numpy()
                            level_idx = None

                        else:
                            levels = np.array(self.atmos_levels)
                            idx_lv = np.argmin(np.abs(levels - target_level))
                            key = vname[0]   # t850 → t, z500 → z

                            pred_v = pred.atmos_vars[key][0, 0, idx_lv].cpu().detach().numpy()
                            gt_v   = label.atmos_vars[key][0, 0, idx_lv].cpu().detach().numpy()
                            level_idx = int(idx_lv)

                        if convert_method:
                            pred_v = convert_method(pred_v)
                            gt_v = convert_method(gt_v)

                        this_date = predict_times[step] + timedelta(hours=self.tranditional_model_leadtime)
                        this_date = this_date.strftime("%Y-%m-%d_%H")

                        fname = f"{self.test_on_map_dir}/{this_date}/{vname}.png"
                        output_path = os.path.dirname(fname)
                        if not os.path.exists(output_path):
                            os.makedirs(output_path, exist_ok=True)

                        self.test_on_map(
                            lat_plot,
                            lon_plot,
                            pred_v,
                            gt_v,
                            title=vname.upper(),
                            fname=fname,
                            config={"title": plot_title},
                            forecast_time=this_date,
                            level_idx=level_idx,
                            source_name="NuWa",
                        )

    def test_on_map(
        self,
        lat,
        lon,
        pred,
        gt,
        title,
        fname,
        vmin=None,
        vmax=None,
        config=None,
        forecast_time=None,
        level_idx=None,
        source_name="NuWa",
    ):

        lat = np.array(lat)
        lon = np.array(lon)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={"projection": ccrs.PlateCarree()})

        # Mirror the subplot title style used in plot_hrrr_zero_shot.py
        if level_idx is None:
            labels = [f"Prediction ({source_name})", f"Ground Truth ({source_name})", "Difference |Pred - GT|"]
        else:
            labels = [
                f"Prediction ({source_name}, level_idx={level_idx})",
                f"Ground Truth ({source_name}, level_idx={level_idx})",
                "Difference |Pred - GT|",
            ]
        extent = [lon.min(), lon.max(), lat.min(), lat.max()] # refine the map to certain region like TW, USA

        for ax in axes:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)

            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.6)
            gl.top_labels = True
            gl.right_labels = False
            gl.bottom_labels = False
            gl.left_labels = True

        diff = np.abs(pred - gt)

        vmin = min(pred.min(), gt.min()) if vmin is None else vmin
        vmax = max(pred.max(), gt.max()) if vmax is None else vmax

        im_pred = axes[0].pcolormesh(lon, lat, pred, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        im_gt   = axes[1].pcolormesh(lon, lat, gt,  cmap="RdBu_r", vmin=vmin, vmax=vmax)
        im_diff = axes[2].pcolormesh(lon, lat, diff, cmap="Greys")

        axes[0].set_title(labels[0])
        axes[1].set_title(labels[1])
        axes[2].set_title(labels[2])

        # Attach colorbars to each subplot so they never overlap the map panels,
        # even when bbox_inches="tight" changes the final crop.
        fig.colorbar(im_pred, ax=axes[0], orientation="horizontal", pad=0.06, fraction=0.046)
        fig.colorbar(im_gt,   ax=axes[1], orientation="horizontal", pad=0.06, fraction=0.046)
        fig.colorbar(im_diff, ax=axes[2], orientation="horizontal", pad=0.06, fraction=0.046)

        plt.suptitle(f"{config['title']} - {forecast_time}", fontsize=18, fontweight="bold")

        # Reserve a bit more bottom margin for horizontal colorbars.
        fig.subplots_adjust(top=0.86, bottom=0.12, wspace=0.15, left=0.02, right=0.98)
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

    def downsample_label(self, label: Batch, scale: float = 0.5) -> Batch:
        def resize_tensor(t: torch.Tensor, new_size, scale: float):
            """
            Resize tensor with anti-aliasing. Supports 3D, 4D, 5D tensors.
            Args:
                t: (B, C, H, W) [surface var] or (B, 1, D, H, W) [aero var]
                ex : [5, 1, 140, 180] and [5, 1, 13, 140, 180]
                scale: scaling factor
            Returns:
                torch.Tensor resized by scale, same device as input
            """

            t_np = t.cpu().numpy()
            shape = t_np.shape

            if t_np.ndim == 4:  # (B, T, H, W)
                B, C, H, W = shape
                new_H, new_W = new_size[scale]
                # new_H, new_W = int(H * scale), int(W * scale)
                t_resized = np.stack([
                    np.stack([resize(t_np[b, c], (new_H, new_W), anti_aliasing=True) for c in range(C)], axis=0)
                    for b in range(B)
                ])
            elif t_np.ndim == 5: # (B, T, C, H, W)
                B, C, D, H, W = shape
                new_H, new_W = new_size[scale]
                # new_H, new_W = int(H * scale), int(W * scale)
                t_resized = np.stack([
                    np.stack([
                        resize(t_np[b, c, d], (new_H, new_W), anti_aliasing=True)
                        for d in range(D)
                    ], axis=0)
                    for b in range(B) for c in range(C)
                ])
                t_resized = t_resized.reshape(B, C, D, new_H, new_W)
            else:
                raise ValueError(f"Unexpected tensor shape {shape}")

            return torch.from_numpy(t_resized).to(t.device).float()

        B, C, H, W = label.surf_vars["2t"].cpu().numpy().shape

        patched_H, patched_W = H // 4, W // 4
        half_H, half_W = math.ceil(patched_H/2), math.ceil(patched_W/2)
        quarter_H, quarter_W = math.ceil(half_H/2), math.ceil(half_W/2)

        new_size = {
            0.25 : [quarter_H * 4, quarter_W * 4], # need to check the size
            0.5 : [half_H * 4, half_W * 4], # need to check the size
        }

        if scale != 0.25 and scale != 0.5 :
            raise "Invalid scale"

        # Downsample surface vars
        surf_vars_low = {k: resize_tensor(v, new_size, scale) for k, v in label.surf_vars.items()}
        # Downsample atmospheric vars
        atmos_vars_low = {k: resize_tensor(v, new_size, scale) for k, v in label.atmos_vars.items()}

        # MAE only calculates over surf and atmos vars, so others are kept the same for now
        return Batch(
            surf_vars=surf_vars_low,
            static_vars=label.static_vars, # we are not using it, so it didnt need downsamples
            atmos_vars=atmos_vars_low,
            metadata=label.metadata,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dictionary_path', type=str, default=None)
    parser.add_argument('--checkpoint_path', type = str, default = None, help='optional for only_test mode (the path needs to .pth) (will use official model if not provided)')
    parser.add_argument('--predict_time', required=True, nargs='+', type=str, help='List of years, e.g. --years 2021 2022 2023 / 202101 202102 202103')
    parser.add_argument('--data_dir', type = str, help = "root dir path of the dataset", required=True)
    args = parser.parse_args()
    trainer = Aurora_trainer(args)
    trainer.main_process()
    print("All Done.")