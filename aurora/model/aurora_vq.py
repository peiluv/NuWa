"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import contextlib
import dataclasses
import warnings
from datetime import timedelta
from functools import partial
from typing import Optional
import math
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from aurora.batch import Batch
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder
from aurora.model.dora import DoRAMode
from aurora.model.swin3d import BasicLayer3D, Swin3DTransformerBackbone

# Import the AuroraVQ from CodeFormer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../CodeFormer/basicsr/archs'))
from vqgan_arch import AuroraVQ

# Import the original Aurora classes
from aurora.model.aurora import Aurora, AuroraSmall

from .dconv_vq import VQDConv
from einops import rearrange
import time

__all__ = ["AuroraWithVQ", "AuroraSmallWithVQ"]


class AuroraWithVQ(Aurora):
    """The Aurora model with integrated Vector Quantization.

    This version inherits from the original Aurora class and adds VQ layers
    between the encoder and backbone.
    """

    def __init__(
        self,
        # VQ-specific parameters
        codebook_size: int = 512,
        vq_sigma: float = 0.25,
        # VQDConv parameters
        use_vq_dconv: bool = False,
        vq_dconv_atoms: int = 128, # codebook size
        # RD-module alignment: RD weighted mixing coefficient
        lambda_rd: float = 0.8,  # RD weight for baseline X, default 0.5 (fair initialization for balanced start)
        # Lambda Generator parameters
        enable_lambda_generator: bool = True,  # Enable dynamic lambda_rd generation
        lambda_generator_type: str | None = None,  # 'mlp' / 'conv' / None
        # Pass all other parameters to parent Aurora class
        freeze_stage2: bool = False,
        **kwargs
    ) -> None:
        """Construct an instance of the Aurora model with VQ.

        RD-module alignment: F = λ_RD·X + (1-λ_RD)·Z_VQ
        where X is encoder output (baseline) and Z_VQ is dictionary feature (D_out).
        """
        # Initialize parent Aurora class first
        super().__init__(**kwargs)

        # Store VQ parameters
        self.codebook_size = codebook_size
        self.vq_sigma = vq_sigma
        self.use_vq_dconv = use_vq_dconv
        self.vq_dconv_atoms = vq_dconv_atoms
        self.freeze_stage2 = freeze_stage2
        self.lambda_generator_type = lambda_generator_type

        # Learnable scaling factor for RD mixing (lambda_rd)
        # Initialize with s_init = logit(lambda_rd) so that sigmoid(s_init) == lambda_rd
        _lambda_rd = float(lambda_rd)
        _lambda_rd = max(1e-6, min(1.0 - 1e-6, _lambda_rd)) # 1e-6 ~ 1
        s_init = torch.logit(torch.tensor(_lambda_rd, dtype=torch.float32))
        self.lambda_rd = nn.Parameter(s_init.reshape(1))

        # Initialize the 3D Vector Quantizer (Not using in forward)
        self.aurora_vq = AuroraVQ(
            codebook_size=codebook_size,
            emb_dim=self.encoder.embed_dim,  # Use encoder's embed_dim
            sigma=vq_sigma,
        )

        # MLP
        


        # 2026/1/21 code review can remove

        # Add normalization layer only
        # Pre-VQ normalization (Not using in forward)
        # 1st stage
         # 2026/1/21 code review can remove
        self.pre_vq_norm = torch.nn.LayerNorm(self.encoder.embed_dim, eps=1e-6)

        # NOTE: The learnable fusion weight parameter is named lambda_rd
        # RD weighted mixing F = λ_RD·X + (1-λ_RD)·Z_VQ provides supervision through
        # reconstruction loss while allowing lambda_rd to adapt during training.

        self.vq_dconv = VQDConv(
            in_channels=self.backbone.embed_dim * 4,
            atoms=vq_dconv_atoms, # codebook size
            enable_lambda_generator=enable_lambda_generator,
            lambda_rd_init=lambda_rd, # initialize lambda generator
            args=self.args
        )

        if enable_lambda_generator:
            print(f"[Info] Use Lambda Generator")
            self.lambda_rd = None
        else :
            print(f"[Info] Use Lambda RD")

        
        self.zvq_project_layer = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

    # def _get_rd_fusion_target_dim(self) -> int:
    #     """Return the feature dim of the RD fusion location (target_dim)."""
        
    #     where_code = getattr(self.args, 'where_code', 'backbone_decoder')

    #     if where_code == "encoder_output":
    #         return int(self.encoder.embed_dim)
    #     elif where_code == "backbone_middle":
    #         num_encoder_layers = len(self.backbone.encoder_layers)
    #         return int(self.backbone.embed_dim * (2 ** (num_encoder_layers - 1)))
    #     elif where_code == "backbone_decoder":
    #         return int(self.backbone.embed_dim)
    #     else:
    #         # Keep backward compatibility: unknown values fall back to backbone_decoder
    #         return int(self.backbone.embed_dim)


    # def _build_zvq_project_layer(self) -> None:

    #     vq_output_dim = int(self.encoder.embed_dim)
    #     target_dim = int(self._get_rd_fusion_target_dim())
    #     self._fusion_target_dim = target_dim

    #     if vq_output_dim == target_dim:
    #         # Scenario A: dims match, still enforce d -> 2d -> d
    #         hidden_dim = 2 * vq_output_dim
    #         self.zvq_project_layer = nn.Sequential(
    #             nn.Linear(vq_output_dim, hidden_dim),
    #             nn.GELU(),
    #             nn.Linear(hidden_dim, target_dim),
    #         )
    #     else:
    #         # Scenario B: dims differ, enforce 2-layer with hidden ~= geometric mean
    #         hidden_dim = int((vq_output_dim * target_dim) ** 0.5)
    #         hidden_dim = max(1, self._round_to_multiple(hidden_dim, multiple=64))
    #         self.zvq_project_layer = nn.Sequential(
    #             nn.Linear(vq_output_dim, hidden_dim),
    #             nn.GELU(),
    #             nn.Linear(hidden_dim, target_dim),
    #         )

    def forward(self, batch: Batch, leadtime: int = 24, batch_idx = 0) -> tuple[Batch, torch.Tensor, dict]:
        """Forward pass with VQ integration."""
        # Get the first parameter. We'll derive the data type and device from this parameter.
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)

        H, W = batch.spatial_shape

        # Insert batch and history dimension for static variables.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        )

        lead_time_td = timedelta(hours=leadtime)

        # 1. Aurora Perceiver3DEncoder output (inherited from parent)
        # X: baseline feature from encoder
        latent_feat = self.encoder(
            batch,
            lead_time=lead_time_td,
        )  # Shape: (B, L, D) - This is X (baseline)

        # 2. VQDConv：Encoder → VQDConv
        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )
        c1, h, w = patch_res
        x_reshaped = rearrange(latent_feat, 'b (h w c1) c2 -> b (c1 c2) h w', c1=c1, h=h, w=w)

        if self.args.count_time :
            start = time.perf_counter()


        vq_out = self.vq_dconv(x_reshaped, batch_idx=batch_idx)

        if self.args.count_time :
            end = time.perf_counter()
            print(f"prepare batch dataset: {end - start:.6f} 秒")



        if isinstance(vq_out, (tuple, list)):
            if len(vq_out) >= 4:
                prior, r_cg, dconv_stats, lambda_rd_dynamic, entropy_loss, valid_count = vq_out
            elif len(vq_out) >= 3:
                prior, r_cg, dconv_stats, entropy_loss = vq_out
                lambda_rd_dynamic = None
            elif len(vq_out) >= 2:
                prior, r_cg, entropy_loss = vq_out
                dconv_stats = None
                lambda_rd_dynamic = None
            else:
                prior, r_cg, dconv_stats, lambda_rd_dynamic, entropy_loss = vq_out, None, None, None
        else:
            prior, r_cg, dconv_stats, lambda_rd_dynamic, entropy_loss = vq_out, None, None, None

        z_vq = rearrange(prior, 'b (c1 c2) h w -> b (h w c1) c2', c1=c1)
        # [3, 30284, 256]
        if lambda_rd_dynamic is not None: # have lambda generator
            lambda_rd_to_use = lambda_rd_dynamic
            # print(f"[Info] in aurora vq, lamda_rd_to_use is from lambda generator")
        elif hasattr(self, 'lambda_rd') and self.lambda_rd is not None:
            lambda_rd_to_use = self.lambda_rd
            # print(f"[Info] in aurora vq, lamda_rd_to_use is from lambda rd")
        else:
            lambda_rd_to_use = None

        ### 
             # 2026/1/21 code review
             # wait for projection layer
        ###
        if self.args.use_zvq_project_layer:
            z_vq = self.zvq_project_layer(z_vq)
            # print(f"z_vq.shape = {z_vq.shape}")

        self.last_encoder_output = latent_feat
        self.last_dict_output = z_vq
        vq_loss = torch.tensor(0.0, device=latent_feat.device)
        vq_stats = None

        # Store lambda_rd for analysis (convert to scalar if needed)
        lambda_rd_scalar = None
        if lambda_rd_to_use is not None:
            if isinstance(lambda_rd_to_use, torch.Tensor):
                if lambda_rd_to_use.dim() > 0:
                    # For dynamic lambda_rd, compute mean to get representative value
                    lambda_rd_scalar = torch.sigmoid(lambda_rd_to_use).mean().item()
                else:
                    lambda_rd_scalar = torch.sigmoid(lambda_rd_to_use).item()
            else:
                lambda_rd_scalar = float(lambda_rd_to_use)


        """
            dconv_stats -> vq_stats
        """
        try:
            if isinstance(dconv_stats, dict) and ('indices_hw' in dconv_stats):
                idx_hw = dconv_stats['indices_hw']  # (B, h, w)
                idx_expand = idx_hw[:, None, :, :].repeat(1, c1, 1, 1)   # (B, c1, h, w)
                indices_tokens = rearrange(idx_expand, 'b c1 h w -> b 1 (h w c1)')  # (B, 1, L)
                vq_stats = {
                    'indices': indices_tokens.to(torch.int32),
                    'avg_probs': dconv_stats.get('avg_probs', torch.tensor(0.0, device=latent_feat.device)),
                    'max_probs': dconv_stats.get('max_probs', torch.tensor(0.0, device=latent_feat.device)),
                }
            else:
                vq_stats = {}

            # Add lambda_rd scalar to vq_stats for analysis
            if lambda_rd_scalar is not None:
                vq_stats['lambda_rd'] = lambda_rd_scalar
        except Exception:
            vq_stats = None

        # 4. Pass encoder output (latent_feat) to Swin3DTransformerBackbone
        # RD fusion will be performed INSIDE the backbone at the last decoder layer
        # (after decoder processing, before final concatenation)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext():
            x, multi_latents = self.backbone(
                latent_feat,  # Pass encoder output directly (not fused)
                lead_time=lead_time_td,
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
                z_vq=z_vq,  # Pass VQDConv output for RD fusion inside backbone
                lambda_rd=lambda_rd_to_use,  # Pass dynamic or fixed fusion weight
                freeze_stage2=self.freeze_stage2
            )

        # 6. Aurora Perceiver3DDecoder receives backbone output for reconstruction (inherited from parent)
        pred = self.decoder(
            x,
            batch,
            lead_time=lead_time_td,
            patch_res=patch_res,
        )

        # Remove batch and history dimension from static variables.
        pred = dataclasses.replace(
            pred,
            static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
        )

        # Insert history dimension in prediction. The time should already be right.
        pred = dataclasses.replace(
            pred,
            surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
            atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
        )

        # Superes: Also decode intermediate latents (skip the last one, same as x)
         # ---- multi-resolution branch (GFP-GAN style) ----
        C_full, Hp_full, Wp_full = patch_res
        L_full = C_full * Hp_full * Wp_full

        # print(f"in Aurora Model forward, multi_latents len = {len(multi_latents)}")
        multi_preds = []

        latent_patch_res_list = [
            (C_full, math.ceil(math.ceil(Hp_full/2)/2), math.ceil(math.ceil(Wp_full/2)/2)),
            (C_full, math.ceil(Hp_full/2), math.ceil(Wp_full/2)),
        ]

        for i, latent in enumerate(multi_latents):  # use only the first one
            # print("latent.shape =", latent.shape)

            d_1_2 = latent.shape[2] // 2

            latent_x = latent[..., :d_1_2]
            latent_skip = latent[..., d_1_2:]


            latent_x = self.mr_proj_layers[i][0](latent_x)
            latent_skip = self.mr_proj_layers[i][1](latent_skip)

            latent = torch.cat([latent_x, latent_skip], dim=-1)


            # print("latent.shape after projection =", latent.shape)


            B, L_low, D = latent.shape
            # print(f"full patch_res = {patch_res}")
            # print(f"L_full = {L_full}")

            # infer spatial scale from token ratio
            ratio = L_full // L_low           # e.g. 118272 / 29568 = 4
            scale = int(math.sqrt(ratio))     # → 2 (so spatial scale = 1/2)

            # assert scale * scale == ratio, f"Non-square scale factor: ratio={ratio}"

            Hp_low = Hp_full // scale         # 132 / 2 = 66
            Wp_low = Wp_full // scale         # 224 / 2 = 112

            latent_patch_res = latent_patch_res_list[i]

            # print(f"latent_patch_res = {latent_patch_res}")

            pred_low = self.decoder(
                latent,
                batch, # same as original resolution??
                lead_time=timedelta(hours=6),
                patch_res=latent_patch_res,    # how many patches in this resolution
            )

            # same post-processing as you already do
            pred_low = dataclasses.replace(
                pred_low,
                static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
            )
            pred_low = dataclasses.replace(
                pred_low,
                surf_vars={k: v[:, None] for k, v in pred_low.surf_vars.items()},
                atmos_vars={k: v[:, None] for k, v in pred_low.atmos_vars.items()},
            )
            # pred_low = pred_low.unnormalise(surf_stats=self.surf_stats)

            # print(f"Intermediate pred {i} surf_vars['2t'].shape =", pred_low.surf_vars["2t"].shape)
            multi_preds.append(pred_low)


        # vq_loss absolutely == 0
        #

        return pred, multi_preds, vq_loss, vq_stats, entropy_loss, valid_count

    def load_vq_codebook(self, codebook_path: str):
        if not hasattr(self, 'vq_dconv'):
            raise ValueError("VQDConv is not enabled/initialized. Set use_vq_dconv=True.")

        codebook = torch.load(codebook_path, map_location='cpu')
        if isinstance(codebook, dict):
            if 'aurora_vq.embedding.weight' in codebook:
                codebook_tensor = codebook['aurora_vq.embedding.weight']
            elif 'model_state_dict' in codebook:
                state_dict = codebook['model_state_dict']
                if 'aurora_vq.embedding.weight' in state_dict:
                    codebook_tensor = state_dict['aurora_vq.embedding.weight']
                else:
                    raise KeyError("VQ codebook not found in checkpoint")
            else:
                raise KeyError("Invalid checkpoint format")
        else:
            codebook_tensor = codebook

        emb = codebook_tensor
        shape = tuple(emb.shape)

        if len(shape) == 3 and shape[0] == 1:
            emb = emb.reshape(shape[1], shape[2])
        elif len(shape) == 2:
            if shape[1] == self.codebook_size and shape[0] != self.codebook_size:
                emb = emb.t().contiguous()
        else:
            raise ValueError(f"Unsupported codebook tensor shape: {shape}")

        if emb.shape[0] != self.codebook_size:
            raise ValueError(
                f"Codebook size mismatch: expected {self.codebook_size}, got {emb.shape[0]}"
            )

        self.vq_dconv.load_codebook_weights(emb)
        print(f"Successfully loaded VQ codebook from {codebook_path}")

    # Allen remove some method because they can be inherited

# Pre-configured variants
AuroraSmallWithVQ = partial(
    AuroraWithVQ,
    # 繼承 AuroraSmall
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
    use_lora=False,
    codebook_size=512,
    vq_sigma=0.25,
    lambda_rd=0.8,  # RD-module alignment: default weighted mixing coefficient (fair initialization)
)

AuroraHighResWithVQ = partial(
    AuroraWithVQ,
    patch_size=10,
    # 繼承 AuroraSmall
    encoder_depths=(6, 8, 8),
    decoder_depths=(8, 8, 6),
    codebook_size=512,
    vq_sigma=0.25,
    lambda_rd=0.8,  # RD-module alignment: default weighted mixing coefficient (fair initialization)
)