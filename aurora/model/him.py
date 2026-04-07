## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import numbers
from einops import rearrange
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
##########################################################################

    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # x=rearrange(x,"b c h w -> b h w c") #????
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias



# w/o shape
class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)
    
# class BasicLayer(nn.Module):
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, num_blocks, group):

#         super().__init__()
#         self.group = group

#         # build blocks
#         self.blocks = nn.ModuleList([TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
#                                     bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in range(num_blocks)])
#         if self.group > 1:
#             self.him = HIM(dim, num_heads, bias, embed_dim, LayerNorm_type)

#     def forward(self, x, prior=None):
#         if prior is not None and self.group > 1:
#             x = self.him(x, prior)
#             prior = None

#         for blk in self.blocks:
#             x = blk(x, prior)
                
#         return x

def auto_pad(kernel_size: _size_2_t, dilation: _size_2_t = 1, **kwargs) -> Tuple[int, int]:
    """
    Auto Padding for the convolution blocks
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)

def create_activation_function(activation: str) -> nn.Module:
    """
    Retrieves an activation function from the PyTorch nn module based on its name, case-insensitively.
    """
    if not activation or activation.lower() in ["false", "none"]:
        return nn.Identity()

    activation_map = {
        name.lower(): obj
        for name, obj in nn.modules.activation.__dict__.items()
        if isinstance(obj, type) and issubclass(obj, nn.Module)
    }
    if activation.lower() in activation_map:
        return activation_map[activation.lower()](inplace=True)
    else:
        raise ValueError(f"Activation function '{activation}' is not found in torch.nn")
    
# ----------- Basic Class ----------- #
class Conv(nn.Module):
    """A basic convolutional block that includes convolution, batch normalization, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        *,
        activation: Optional[str] = "SiLU",
        **kwargs,
    ):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2)
        self.act = create_activation_function(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

class DConv(nn.Module):
    def __init__(self, in_channels, alpha=0.8, atoms=64):
        super().__init__()

        self.CG = Conv(in_channels, atoms, 1)
        self.GIE = Conv(atoms, atoms, 5, groups=atoms, activation=False)
        self.D = Conv(atoms, in_channels, 1, activation=False)

    def PONO(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-5)
        return x

    def forward(self, r):
        # print(r.shape) # 16,35,45,1024
        x = self.CG(r)
        x = self.GIE(x)
        x = self.PONO(x)
        x = self.D(x)
        return x # self.alpha * x + (1 - self.alpha) * r

## Hierarchical Integration Module
class HIM(nn.Module):
    def __init__(self, dim, num_heads=8, embed_dim=256, bias=True, LayerNorm_type="BiasFree", qk_scale=None):
        super(HIM, self).__init__()
        # self.get_prior = DConv(in_channels=dim)
        self.raw_alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
        self.norm2 = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)

        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(embed_dim*4, 2*dim, bias=bias)
        
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, prior):
        B, C, H, W = x.shape
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        prior = rearrange(prior, 'b c h w -> b (h w) c')
        _x = self.norm1(x)
        prior = self.norm2(prior)

        # _x = rearrange(_x, 'b h w c -> b (h w) c')
        # prior = rearrange(prior, 'b h w c -> b (h w) c')

        q = self.q(prior)
        kv = self.kv(_x)
        k,v = kv.chunk(2, dim=-1)   

        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head n c -> b n (head c)', head=self.num_heads)
        out = self.proj(out)
        
        # sum
        alpha = torch.sigmoid(self.raw_alpha)
        x = (1 - alpha)*x + alpha * out
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x

# def main():
#     # x =torch.rand((16, 35, 45, 1024)) # B,H,W,C
#     # x=torch.rand((16,6300,256))
#     # B, HWC1, C2 = x.shape
#     # C = C2*4
#     # H=35
#     # W=45
#     # x=rearrange(x, 'b (hwc1) c2 -> b c h w', c=C, h=H, w=W)
#     dictionary = torch.tensor(np.load(f'/home/nccu/Tina/repo1/dictionary/64_dictionary_202101.npy'), dtype=torch.float32)
#     dictionary = dictionary.permute(1, 0)   # (1024, 64)
#     dictionary = dictionary.unsqueeze(2)    # (1024, 64, 1)
#     dictionary = dictionary.unsqueeze(3)    # (1024, 64, 1, 1)
#     print(dictionary.shape)
#     x =torch.rand((16, 1024, 35, 45))
#     get_prior = DConv(in_channels=x.shape[1])
#     print(get_prior.D.conv.weight.shape)
#     prior = get_prior(x)
#     model = HIM(dim = x.shape[1])
#     pred = model(x, prior)
#     print(pred.shape)


# if __name__ == "__main__":
#     main()