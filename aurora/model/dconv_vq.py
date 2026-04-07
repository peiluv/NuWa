import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from importlib import import_module
import os

class VQDConv(nn.Module):
    def __init__(self, in_channels, atoms=256, codebook_projection=None,
                 enable_lambda_generator=True, lambda_rd_init=0.8, args=None):
        """
        Args:
            in_channels: input channel dimension
            atoms: codebook size (number of atoms)
            codebook_projection: optional projection layer for codebook (e.g., 512->256)
            enable_lambda_generator: whether to enable dynamic lambda_rd generator
            lambda_rd_init: initial value for lambda_rd
        """
        super().__init__()
        self.args = args
        self.atoms = atoms
        self.in_channels = in_channels

        # kernel size == 1 => won't change the h, w
        # Channel compression: 1024 -> 256
        self.channel_compression = nn.Conv2d(in_channels, 256, 1)

        # Coefficient Generator (CG) - project to "atoms" dimension
        self.CG = nn.Conv2d(256, atoms, 1)

        # Group-wise Information Extraction (GIE)
        self.GIE = nn.Conv2d(atoms, atoms, 5, groups=atoms, padding=2) # b, a, h, w (atom rate)

        # Codebook (256-dim embedding space)
        self.D = nn.Conv2d(atoms, 256, 1, bias=False) # b, 256, h, w # isn't this kernel the rate?

        # Channel expansion: 256 -> 1024 (restore original input channels)
        self.channel_expansion = nn.Conv2d(256, in_channels, 1)

        # Residual prediction head: from 256-dim features to residual R_CG with same shape as input
        # [Note] r_cg is currently unused and commented out to reduce training cost
        # self.residual_head = nn.Conv2d(256, in_channels, 1)

        # Dynamic projection layer (for 13B codebook 512->256 projection)
        self.codebook_projection = codebook_projection
        self.original_codebook = None  # store original 512-dim codebook (if dynamic projection is used)

    def PONO(self, x):
        """Position Normalization - consistent with Dictionary-based modules"""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-5)
        return x

    def _apply_weight_normalization(self):
        """
        Apply explicit L2 / weight normalization (WN) to dictionary atoms.

        Each atom corresponds to one input channel of D:
          - D.weight: shape (out_channels=256, in_channels=atoms, 1, 1)
          - For atom i, its vector is D.weight[:, i, 0, 0]
        We normalise every such vector to unit length:
          alpha_i <- alpha_i / ||alpha_i||
        This is equivalent to WN(D) in YOLO-RD, and is applied at every forward.
        """
        with torch.no_grad():
            w = self.D.weight  # (C_out, C_in, 1, 1)
            # reshape to (C_out, C_in)
            w_2d = w.view(w.shape[0], w.shape[1]) # embed_dim, n_codes
            # L2 norm over output-dim per atom (per input channel)
            norms = torch.norm(w_2d, dim=0, keepdim=True).clamp_min(1e-6)
            w_2d = w_2d / norms
            self.D.weight.copy_(w_2d.view_as(w))

    def load_codebook_weights(self, codebook_tensor, codebook_projection=None):
        """Load pretrained codebook weights into convolution D.

        Args:
            codebook_tensor: Codebook tensor of shape (n_e, e_dim)
            codebook_projection: Optional projection layer for dynamic projection (e.g., 512->256)
        """
        if codebook_tensor is None:
            raise ValueError("codebook_tensor is None")

        # Support shapes (n_e, e_dim) or (1, n_e, e_dim)
        if codebook_tensor.dim() == 3 and codebook_tensor.size(0) == 1:
            codebook_tensor = codebook_tensor.squeeze(0)

        if codebook_tensor.dim() != 2:
            raise ValueError(
                f"Unsupported codebook tensor shape: {tuple(codebook_tensor.shape)}."
            )

        n_codes, embed_dim = codebook_tensor.shape
        expected_embed_dim = self.channel_compression.out_channels  # 256

        # If codebook is 512-dim and a projection layer is provided, use dynamic projection mode
        if embed_dim == 512 and codebook_projection is not None:
            # Save original 512-dim codebook and projection layer for dynamic projection
            self.original_codebook = codebook_tensor.clone().detach()
            self.codebook_projection = codebook_projection
            # Initial projection used for sanity check
            with torch.no_grad():
                codebook_tensor = codebook_projection(codebook_tensor) # 512 (aurora big) -> 256 (aurora small)
            embed_dim = codebook_tensor.shape[1]
        else:
            # Static load mode: use codebook directly
            self.original_codebook = None
            self.codebook_projection = None

        # Adjust number of atoms to match self.atoms
        if n_codes != self.atoms:

            if n_codes > self.atoms:
                codebook_tensor = codebook_tensor[:self.atoms]
                if self.original_codebook is not None:
                    self.original_codebook = self.original_codebook[:self.atoms]
            else:
                repeat_times = self.atoms // n_codes
                remainder = self.atoms % n_codes
                repeated = codebook_tensor.repeat(repeat_times, 1)
                if remainder > 0:
                    repeated = torch.cat([repeated, codebook_tensor[:remainder]], dim=0)
                codebook_tensor = repeated
                if self.original_codebook is not None:
                    orig_repeated = self.original_codebook.repeat(repeat_times, 1)
                    if remainder > 0:
                        orig_repeated = torch.cat([orig_repeated, self.original_codebook[:remainder]], dim=0)
                    self.original_codebook = orig_repeated

        # Ensure embedding_dim matches channel_compression output dim (256)
        if embed_dim != expected_embed_dim:
            raise ValueError(
                f"Codebook embedding dim {embed_dim} does not match expected {expected_embed_dim}."
            )

        # Transpose to (out_channels=256, in_channels=self.atoms)
        conv_weight = codebook_tensor.t().contiguous().unsqueeze(-1).unsqueeze(-1) # dim, n

        if conv_weight.shape[0] != self.D.out_channels or conv_weight.shape[1] != self.D.in_channels:
            raise ValueError(
                f"Prepared conv weight shape {tuple(conv_weight.shape)} does not match D weight {tuple(self.D.weight.shape)}."
            )

        with torch.no_grad():
            self.D.weight.copy_(conv_weight) # dim, n_codes, 1, 1
            # If using dynamic projection, D.weight will be updated at forward time
            # If using static loading, D.weight remains frozen
            if self.original_codebook is None:
                # Static mode: freeze codebook (retrieval only, no training)
                self.D.weight.requires_grad = False
            else:
                # Dynamic projection mode: D.weight is dynamically updated in forward,
                # but gradients propagate through the projection layer, not D.weight itself
                self.D.weight.requires_grad = False

    def forward(self, x, batch_idx, spatial_size=None, time_steps=None):
        """
        VQDConv forward pass - aligned with Dictionary-based RD-style modules.
        Input:  (B, 1024, H, W)
        Output: tuple(prior, r_cg, stats, lambda_rd)
          - prior:    (B, 1024, H, W) dictionary prior P (actually used)
          - r_cg:     None (residual head disabled to reduce training cost)
          - stats:    dict with statistics (indices, probabilities, etc.)
          - lambda_rd: dynamically generated fusion weight (if lambda_generator enabled)
        """
        # Channel compression: 1024 -> 256
        x_compressed = self.channel_compression(x)

        # Dynamically generate lambda_rd (using same features as CG)
        lambda_rd = None

        # Dictionary-style retrieval pipeline
        x = self.CG(x_compressed)        # 256 -> atoms
        x = self.GIE(x)                  # depthwise conv over atoms
        x = self.PONO(x)                 # position-wise normalization

        coeff = x                     # (B, atoms, H, W)
        probs = torch.softmax(coeff, dim=1)

        entropy_loss = 0.0
        valid_count = 0.0

        # entropy first and select % next (teacher said)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)  # (B, H, W)
        entropy_loss = entropy.mean()

        B, C, H, W = probs.shape # C means atoms

        # select top k % atoms

        k = max(1, int(C * 0.2)) # valid_atom_rate
        topk_vals, topk_idx = torch.topk(probs, k=k, dim=1)

        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_idx, 1.0) # dim, index, src
        x = x * mask
        probs = probs * mask
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-9)



        # random sample a prob distibution to find how many atoms are used
        b = torch.randint(0, B, (1,)).item()
        h = torch.randint(0, H, (1,)).item()
        w = torch.randint(0, W, (1,)).item()
        pixel_probs = probs[b, :, h, w]   # shape: (C,)
        sorted_probs, sorted_indices = torch.sort(pixel_probs, descending=True)
        valid_count = (sorted_probs > 1e-4).sum().item()

        # Build indices stats (argmax over softmax probabilities used only for logging)
        with torch.no_grad():
            indices = probs.argmax(dim=1).to(torch.int32)  # 3, 67, 113
            avg_probs = probs.mean()
            max_probs = probs.amax(dim=1).mean()

        # Dynamic projection mode: if using 13B codebook, project each forward
        if self.original_codebook is not None and self.codebook_projection is not None:
            # Dynamic projection: project 512-dim codebook to 256-dim
            projected_codebook = self.codebook_projection(self.original_codebook)
            # Convert to convolution weight format and update D.weight

             # 2026/1/21 code review Be careful
            conv_weight = projected_codebook.t().contiguous().unsqueeze(-1).unsqueeze(-1)
            self.D.weight.data = conv_weight

        # Explicit Weight Normalization: ensure each atom (dictionary filter) has unit length
        # Apply L2 normalisation over each input-channel vector of D.weight
        self._apply_weight_normalization()

        x = self.D(x)                    # atoms -> 256 (weighted sum over codebook atoms)
        # x : b, a, h, w : rate, D : (out(dim), in(n))

        # Channel expansion: 256 -> 1024 (restore original input channels)
        prior = self.channel_expansion(x) # b, 1024, h, w

        # Residual prediction (same shape as prior, for possible L_res supervision)
        # [Note] Currently unused and kept as None for API compatibility
        # r_cg = self.residual_head(x)
        r_cg = None  # 返回 None 以保持接口兼容性

        stats = {
            'indices_hw': indices,       # (B, H, W)
            'avg_probs': avg_probs,      # scalar tensor
            'max_probs': max_probs,      # scalar tensor
        }
        # Return lambda_rd as the fourth value (for RD fusion downstream)
        return prior, r_cg, stats, lambda_rd, entropy_loss, valid_count