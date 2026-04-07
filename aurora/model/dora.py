import torch
import torch.nn.functional as F
from torch import nn
"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import math
from typing import Literal, Tuple

import torch
from torch import nn

__all__ = ["DoRA", "DoRARollout", "DoRAMode"]

DoRAMode = Literal["single", "all"]


class DoRA(nn.Module):
    """DoRA adaptation for a linear layer."""

    def __init__(
        self,
        base_layer: nn.Module,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: int = 1,
        dropout: float = 0.0,
    ):
        """Initialise.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            r (int, optional): Rank. Defaults to `4`.
            alpha (int, optional): Alpha. Defaults to `1`.
            dropout (float, optional): Drop-out rate. Defaults to `0.0`.
        """
        super().__init__()

        assert r > 0, "The rank must be strictly positive."
        self.lora_alpha = alpha
        self.r = r

        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))
        self.scaling = self.lora_alpha / self.r

        self.init_weights()

        # init dora magnitude
        self.base_layer = base_layer
        self.base_weight = base_layer.weight
        self.weight_norm= self.get_weight_norm()
        self.dora_weight = nn.Parameter(self.weight_norm, requires_grad=True)


    def init_weights(self) -> None:
        """Initialise weights."""
        # Initialise A the same way as the default for `nn.Linear` and set B to zero.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA adaptation.

        Args:
            x (torch.Tensor): Input to the linear layer.

        Returns:
            torch.Tensor: Additive correction for the output of the linear layer.
        """
        magnitude = self.dora_weight
        weight_norm = self.get_weight_norm()
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        lora_result = self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        base_result = self.base_layer(x)
        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * self.scaling
        return result_dora
    
    def get_weight_norm(self) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        # weight = transpose(weight, self.fan_in_fan_out)
        adapter_weight = self.lora_B@self.lora_A
        weight = self.base_weight + self.scaling * adapter_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        
        return weight_norm

class DoRA_LoHa(nn.Module):
    """DoRA adaptation for a linear layer."""

    def __init__(
        self,
        base_layer: nn.Module,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: int = 1,
        dropout: float = 0.0,
    ):
        """Initialise.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            r (int, optional): Rank. Defaults to `4`.
            alpha (int, optional): Alpha. Defaults to `1`.
            dropout (float, optional): Drop-out rate. Defaults to `0.0`.
        """
        super().__init__()

        assert r > 0, "The rank must be strictly positive."
        self.lora_alpha = alpha
        self.r = r

        self.loha_dropout = nn.Dropout(dropout)

        # loha parameters
        self.loha_w1_a = nn.Parameter(torch.empty(in_features, r))
        self.loha_w1_b = nn.Parameter(torch.empty(r, out_features))
        self.loha_w2_a = nn.Parameter(torch.empty(in_features, r))
        self.loha_w2_b = nn.Parameter(torch.empty(r, out_features))
        self.scaling = self.lora_alpha / self.r

        self.init_weights()

        # init dora magnitude
        self.base_layer = base_layer
        self.base_weight = base_layer.weight
        self.weight_norm= self.get_weight_norm()
        self.dora_weight = nn.Parameter(self.weight_norm, requires_grad=True)


    def init_weights(self) -> None:
        """Initialise weights."""
        nn.init.kaiming_uniform_(self.loha_w1_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.loha_w1_b, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.loha_w2_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.loha_w2_b, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA adaptation.

        Args:
            x (torch.Tensor): Input to the linear layer.

        Returns:
            torch.Tensor: Additive correction for the output of the linear layer.
        """
        magnitude = self.dora_weight
        weight_norm = self.get_weight_norm()
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        loha_result=self.loha_dropout(x) @ ((self.loha_w1_a @ self.loha_w1_b) * (self.loha_w2_a @ self.loha_w2_b))
        base_result = self.base_layer(x)
        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * loha_result * self.scaling
        return result_dora
    
    def get_weight_norm(self) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        adapter_weight = (self.loha_w1_a @ self.loha_w1_b) * (self.loha_w2_a @ self.loha_w2_b)
        weight = self.base_weight + self.scaling * adapter_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        
        return weight_norm

class DoRA_LoKr(nn.Module):
    """DoRA adaptation for a linear layer."""
    def __init__(
        self,
        base_layer: nn.Module,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: int = 1,
        dropout: float = 0.0,
    ):
        """Initialise.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            r (int, optional): Rank. Defaults to `4`.
            alpha (int, optional): Alpha. Defaults to `1`.
            dropout (float, optional): Drop-out rate. Defaults to `0.0`.
        """
        super().__init__()

        assert r > 0, "The rank must be strictly positive."
        self.alpha = alpha
        self.r = r

        self.lokr_dropout = nn.Dropout(dropout)
        in_m, in_n = self.factorization(in_features, -1)
        out_l, out_k = self.factorization(out_features, -1)
        # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d
        shape = ((out_l, out_k), (in_m, in_n))


        # lokr parameters
        self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], r))
        self.lokr_w1_b = nn.Parameter(torch.empty(r, shape[1][0]))
        self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], r))
        self.lokr_w2_b = nn.Parameter(torch.empty(r, shape[1][1]))
        self.scaling = self.alpha / self.r

        self.init_weights()

        # init dora magnitude
        self.base_layer = base_layer
        self.base_weight = base_layer.weight
        self.weight_norm= self.get_weight_norm()
        self.dora_weight = nn.Parameter(self.weight_norm, requires_grad=True)


    def init_weights(self) -> None:
        """Initialise weights."""
        nn.init.zeros_(self.lokr_w1_a)
        nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lokr_w2_b, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA adaptation.

        Args:
            x (torch.Tensor): Input to the linear layer.

        Returns:
            torch.Tensor: Additive correction for the output of the linear layer.
        """
        magnitude = self.dora_weight
        weight_norm = self.get_weight_norm()
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        w1 = self.lokr_w1_a @ self.lokr_w1_b
        w2 = self.lokr_w2_a @ self.lokr_w2_b
        lokr_result= self.lokr_dropout(x) @ torch.kron(w1.transpose(0, 1), w2.transpose(0, 1))
        base_result = self.base_layer(x)
        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lokr_result * self.scaling
        return result_dora
    
    def get_weight_norm(self) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        w1 = self.lokr_w1_a @ self.lokr_w1_b
        w2 = self.lokr_w2_a @ self.lokr_w2_b
        adapter_weight = torch.kron(w1.transpose(0, 1), w2.transpose(0, 1))
        weight = self.base_weight + self.scaling * adapter_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        
        return weight_norm
    
    @ staticmethod
    def factorization(dimension: int, factor: int = -1) -> Tuple[int, int]:
        """Factorizes the provided number into the product of two numbers

        Args:
            dimension (`int`): The number that needs to be factorized.
            factor (`int`, optional):
                Factorization divider. The algorithm will try to output two numbers, one of each will be as close to the
                factor as possible. If -1 is provided, the decomposition algorithm would try to search dividers near the
                square root of the dimension. Defaults to -1.

        Returns:
            Tuple[`int`, `int`]: A tuple of two numbers, whose product is equal to the provided number. The first number is
            always less than or equal to the second.

        Example:
            ```py
            >>> factorization(256, factor=-1)
            (16, 16)

            >>> factorization(128, factor=-1)
            (8, 16)

            >>> factorization(127, factor=-1)
            (1, 127)

            >>> factorization(128, factor=4)
            (4, 32)
            ```
        """

        if factor > 0 and (dimension % factor) == 0:
            m = factor
            n = dimension // factor
            return m, n
        if factor == -1:
            factor = dimension
        m, n = 1, dimension
        length = m + n
        while m < n:
            new_m = m + 1
            while dimension % new_m != 0:
                new_m += 1
            new_n = dimension // new_m
            if new_m + new_n > length or new_m > factor:
                break
            else:
                m, n = new_m, new_n
        if m > n:
            n, m = m, n
        return m, n

class DoRARollout(nn.Module):
    """Per-roll-out-step LoRA finetuning."""

    def __init__(
        self,
        base_layer: nn.Module,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        max_steps: int = 40,
        mode: DoRAMode = "single",
    ):
        """Initialise.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            r (int, optional): Rank. Defaults to `4`.
            alpha (int, optional): Alpha. Defaults to `1`.
            dropout (float, optional): Drop-out rate. Defaults to `0.0`.
            max_steps (int, optional): Maximum number of roll-out steps. Defaults to `40`.
            mode (str, optional): Mode. `"single"` uses the same LoRA for all roll-out steps,
                and `"all"` uses a different LoRA for every roll-out step. Defaults to `"single"`.
        """
        super().__init__()

        self.mode = mode
        self.max_steps = max_steps
        lora_layers = max_steps if mode == "all" else 1
        self.doras = nn.ModuleList(
            [
                DoRA(base_layer, in_features, out_features, r=r, alpha=alpha, dropout=dropout)
                for _ in range(lora_layers)
            ]
        )

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Compute the LoRA adaptation.

        Args:
            x (torch.Tensor): Input to the linear layer.
            step (int): Roll-out step, starting at zero.

        Returns:
            torch.Tensor: Additive correction for the output of the linear layer.
        """
        assert step >= 0, f"Step must be non-negative, found {step}."

        if step >= self.max_steps:
            return 0

        if self.mode == "single":
            return self.doras[0](x)
        elif self.mode == "all":
            return self.doras[step](x)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

