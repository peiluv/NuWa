import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from datetime import timedelta
from typing import Optional

from aurora.model.swin3d import Swin3DTransformerBackbone

def lead_time_expansion(lead_times, embed_dim):
    """Expand lead times to embedding dimension"""
    return lead_times.unsqueeze(-1).expand(-1, embed_dim)

class Swin3DTransformerBackboneWithVQ(Swin3DTransformerBackbone):
    def __init__(self, use_vq_dconv=False, vq_dconv_atoms=128, **kwargs):
        super().__init__(**kwargs)
        # 按新架構：VQDConv 已移至 Encoder→Backbone 之間，backbone 不再進行任何區域混合
        self.use_vq_dconv = False
        self.vq_dconv_atoms = vq_dconv_atoms

    def forward(self, x, lead_time, patch_res, rollout_step=1):
        # 新架構：Backbone 僅負責主幹表徵學習，不做任何 VQDConv/混合

        # 原有的 Swin3D 處理流程
        all_enc_res, padded_outs = self.get_encoder_specs(patch_res)

        lead_hours = lead_time / timedelta(hours=1)
        lead_times = lead_hours * torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
        c = self.time_mlp(lead_time_expansion(lead_times, self.embed_dim).to(dtype=x.dtype))

        skips = []
        for i, layer in enumerate(self.encoder_layers):
            x, x_unscaled = layer(x, c, all_enc_res[i], rollout_step=rollout_step)
            skips.append(x_unscaled)

        for i, layer in enumerate(self.decoder_layers):
            index = self.num_decoder_layers - i - 1
            x, _ = layer(
                x,
                c,
                all_enc_res[index],
                padded_outs[index - 1],
                rollout_step=rollout_step,
            )

            if 0 < i < self.num_decoder_layers - 1:
                # For the intermediate stages, we use additive skip connections.
                x = x + skips[index - 1]
            elif i == self.num_decoder_layers - 1:
                # For the last stage, we perform concatentation like in Pangu.
                x = torch.cat([x, skips[0]], dim=-1)

        return x

    def load_vq_codebook(self, codebook_path: str):
        # 新架構：codebook 載入由 AuroraWithVQ 外掛的 VQDConv 處理
        raise NotImplementedError("Backbone no longer manages VQDConv codebook. Use AuroraWithVQ.load_vq_codebook.")