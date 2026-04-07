import torch
import torch.nn as nn
import torch.nn.functional as F

# class LambdaGeneratorMLP(nn.Module):
#     """Lambda Generator - Dynamically generates lambda_rd fusion weight

#     Simple MLP: Linear -> GELU -> Linear -> scalar -> sigmoid

#     Design philosophy:
#     - Uses a simple MLP to dynamically generate lambda_rd based on input features
#     - Global decision: entire input shares a single fusion weight
#     - Computationally efficient: Global Average Pooling first, then MLP
#     """

#     def __init__(self, in_channels=256, hidden_dim=64):
#         """
#         Args:
#             in_channels: Input feature channels (recommended: 256 from channel_compressed dimension)
#             hidden_dim: Hidden layer dimension (default: 64)
#         """
#         super().__init__()
#         self.lambda_net = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # (B, C, H, W) -> (B, C, 1, 1)
#             nn.Flatten(),              # (B, C, 1, 1) -> (B, C)
#             nn.Linear(in_channels, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, 1),  # -> (B, 1)
#         )

#     def forward(self, x):
#         """
#         Args:
#             x: Input features (B, C, H, W) - typically channel_compressed features
#         Returns:
#             lambda_rd: Fusion weight, mapped to [0, 1] via sigmoid, shape: (B, 1)
#         """
#         lambda_logit = self.lambda_net(x)  # (B, 1) - output logit
#         lambda_rd = torch.sigmoid(lambda_logit)  # (B, 1) - mapped to [0, 1]
#         return lambda_rd

#     def initialize_with_value(self, target_value=0.8):
#         """Initialize Lambda Generator to output close to target_value initially

#         Args:
#             target_value: Target initial value (typically in [0, 1] range)
#         """
#         with torch.no_grad():
#             # Compute corresponding logit value
#             target_value = max(1e-6, min(1.0 - 1e-6, float(target_value)))
#             bias_init = torch.logit(torch.tensor(target_value, dtype=torch.float32))

#             # Set bias of the last layer
#             last_layer = self.lambda_net[-1]
#             if hasattr(last_layer, 'bias') and last_layer.bias is not None:
#                 last_layer.bias.fill_(bias_init.item())

class LambdaGeneratorConv(nn.Module):
    """Lambda Generator (Per-Location Conv version) - Dynamically generates lambda_rd fusion weight

    Per-location conv head: generates independent lambda_rd for each spatial location

    Design philosophy:
    - Uses 1x1 convolution (similar to CG) to preserve spatial structure
    - Local decision: each spatial location independently determines fusion weight
    - Strong expressive power: can adapt to different requirements of different spatial regions
    """

    def __init__(self, in_channels=256, hidden_dim=64):
        """
        Args:
            in_channels: Input feature channels (recommended: 256 from channel_compressed dimension)
            hidden_dim: Hidden layer dimension (default: 64)
        """
        super().__init__()
        self.lambda_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),  # (B, C, H, W) -> (B, hidden_dim, H, W)
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),  # (B, hidden_dim, H, W) -> (B, 1, H, W)
        )

    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W) - typically channel_compressed features
        Returns:
            lambda_rd: Fusion weight, mapped to [0, 1] via sigmoid, shape: (B, 1, H, W)
        """
        lambda_logit = self.lambda_net(x)  # (B, 1, H, W) - output logit
        lambda_rd = torch.sigmoid(lambda_logit)  # (B, 1, H, W) - mapped to [0, 1]
        return lambda_rd

    def initialize_with_value(self, target_value=0.8):
        """Initialize Lambda Generator to output close to target_value initially

        Args:
            target_value: Target initial value (typically in [0, 1] range)
        """
        with torch.no_grad():
            # Compute corresponding logit value
            target_value = max(1e-6, min(1.0 - 1e-6, float(target_value)))
            bias_init = torch.logit(torch.tensor(target_value, dtype=torch.float32))

            # Set bias of the last layer
            last_layer = self.lambda_net[-1]
            if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                last_layer.bias.fill_(bias_init.item())

class LambdaGenerator(nn.Module):
    """Hybrid Lambda Generator - Dynamically generates lambda_rd considering spatial size and temporal information

    Design philosophy:
    - Uses multi-scale pooling to extract spatial features (preserves multi-scale information)
    - Incorporates conditional information: spatial size, time steps, spatial variability
    - Global decision with context awareness, suitable for handling regional data with different temporal and spatial ranges

    Advantages:
    - Considers spatial size: different sized regions will have different lambda
    - Considers temporal information: different time ranges will have different lambda
    - Preserves multi-scale spatial information: does not completely lose spatial structure
    - Easy to integrate: output is still scalar, compatible with existing architecture
    """

    def __init__(self, in_channels=256, hidden_dim=64, max_spatial_size=128*128, max_time_steps=10):
        """
        Args:
            in_channels: Input feature channels (recommended: 256 from channel_compressed dimension)
            hidden_dim: Hidden layer dimension (default: 64)
            max_spatial_size: Maximum spatial size (for normalization), default: 128*128
            max_time_steps: Maximum time steps (for normalization), default: 10
        """
        super().__init__()

        # Multi-scale feature extraction (similar to Spatial Pyramid Pooling)
        # Uses 1x1, 2x2, 4x4 three scales to preserve spatial information at different scales
        self.pool_sizes = [1, 2, 4]
        self.scale_nets = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Flatten(),
                nn.Linear(in_channels * pool_size * pool_size, hidden_dim),
                nn.GELU(),
            )
            for pool_size in self.pool_sizes
        ])

        # Condition dimensions: spatial size, time steps, spatial variability
        self.condition_dim = 3
        self.max_spatial_size = max_spatial_size
        self.max_time_steps = max_time_steps

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * len(self.pool_sizes) + self.condition_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, spatial_size=None, time_steps=None):
        """
        Args:
            x: Input features (B, C, H, W) - typically channel_compressed features
            spatial_size: Spatial size (number of grid points), shape: (B,) or scalar, optional
            time_steps: Number of time steps, shape: (B,) or scalar, optional
        Returns:
            lambda_rd: Fusion weight, mapped to [0, 1] via sigmoid, shape: (B, 1)
        """
        B = x.shape[0]
        device = x.device

        # 1. Extract multi-scale features
        scale_features = []
        for scale_net in self.scale_nets:
            feat = scale_net(x)  # (B, hidden_dim)
            scale_features.append(feat)
        feat_multiscale = torch.cat(scale_features, dim=1)  # (B, hidden_dim * 3)

        # 2. Compute condition features
        # Spatial size normalization
        if spatial_size is None:
            # If not provided, infer from input features
            H, W = x.shape[2], x.shape[3]
            spatial_size = torch.tensor(H * W, device=device, dtype=torch.float32)
            spatial_size = spatial_size / self.max_spatial_size  # Normalize to [0, 1]
            spatial_size = spatial_size.unsqueeze(0).expand(B)
        else:
            spatial_size = torch.tensor(spatial_size, device=device, dtype=torch.float32)
            if spatial_size.dim() == 0:
                spatial_size = spatial_size.unsqueeze(0).expand(B)
            spatial_size = spatial_size / self.max_spatial_size  # Normalize

        # Time steps normalization
        if time_steps is None:
            # If not provided, assume single time step
            time_steps = torch.ones(B, device=device, dtype=torch.float32) / self.max_time_steps
        else:
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32)
            if time_steps.dim() == 0:
                time_steps = time_steps.unsqueeze(0).expand(B)
            time_steps = time_steps / self.max_time_steps  # Normalize

        # Spatial variability (standard deviation) - measures spatial heterogeneity
        spatial_std = x.std(dim=[2, 3]).mean(dim=1)  # (B,) - spatial variability per sample
        # Normalize spatial variability (assume range in [0, 1])
        spatial_std = torch.clamp(spatial_std, 0, 1)

        # 3. Combine conditions
        conditions = torch.stack([
            spatial_size,
            time_steps,
            spatial_std,
        ], dim=1)  # (B, 3)

        # 4. Fuse features and conditions
        feat_combined = torch.cat([feat_multiscale, conditions], dim=1)  # (B, hidden_dim*3 + 3)

        # 5. Generate lambda
        lambda_logit = self.fusion_net(feat_combined)  # (B, 1)
        lambda_rd = torch.sigmoid(lambda_logit)  # (B, 1)

        return lambda_rd

    def initialize_with_value(self, target_value=0.8):
        """Initialize Lambda Generator to output close to target_value initially

        Args:
            target_value: Target initial value (typically in [0, 1] range)
        """
        with torch.no_grad():
            # Compute corresponding logit value
            target_value = max(1e-6, min(1.0 - 1e-6, float(target_value)))
            bias_init = torch.logit(torch.tensor(target_value, dtype=torch.float32))

            # Set bias of the last layer
            last_layer = self.fusion_net[-1]
            if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                last_layer.bias.fill_(bias_init.item())