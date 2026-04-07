"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import math
from typing import Literal, Tuple

import torch
from torch import nn

__all__ = ["LoKr", "LoKrRollout", "LoKrMode"]

LoKrMode = Literal["single", "all"]

class LoKr(nn.Module):
    """LoKr adaptation for a linear layer."""

    def __init__(
        self,
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
        self.r = r
        self.lokr_dropout = nn.Dropout(dropout)

        # shape
        in_m, in_n = factorization(in_features, -1)
        out_l, out_k = factorization(out_features, -1)
        # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d
        shape = ((out_l, out_k), (in_m, in_n))

        # lokr parameters
        self.lokr_w1_a = nn.Parameter(torch.empty(r, shape[0][0]))
        self.lokr_w1_b = nn.Parameter(torch.empty(shape[1][0], r))
        self.lokr_w2_a = nn.Parameter(torch.empty(r, shape[0][1]))
        self.lokr_w2_b = nn.Parameter(torch.empty(shape[1][1], r))
        self.lokr_alpha = alpha
        self.scaling = self.lokr_alpha / self.r

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        nn.init.zeros_(self.lokr_w1_a)
        nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lokr_w2_b, a=math.sqrt(5))
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoKr adaptation.

        Args:
            x (torch.Tensor): Input to the linear layer.

        Returns:
            torch.Tensor: Additive correction for the output of the linear layer.
        """
        # x=self.lokr_dropout(x) @ ((self.lokr_w1_a @ self.lokr_w1_b) * (self.lokr_w2_a @ self.lokr_w2_b))
        w1 = self.lokr_w1_a.transpose(0, 1) @ self.lokr_w1_b.transpose(0, 1)
        w2 = self.lokr_w2_a.transpose(0, 1) @ self.lokr_w2_b.transpose(0, 1)
        # print(w1.shape)
        # print(w2.shape)
        # print(x.shape)
        # print(torch.kron(w1.transpose(0, 1),w2.transpose(0, 1)).shape)
        x= self.lokr_dropout(x) @ torch.kron(w1.transpose(0, 1), w2.transpose(0, 1))
        
        return x * self.scaling


class LoKrRollout(nn.Module):
    """Per-roll-out-step LoKr finetuning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        max_steps: int = 40,
        mode: LoKrMode = "single",
    ):
        """Initialise.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            r (int, optional): Rank. Defaults to `4`.
            alpha (int, optional): Alpha. Defaults to `1`.
            dropout (float, optional): Drop-out rate. Defaults to `0.0`.
            max_steps (int, optional): Maximum number of roll-out steps. Defaults to `40`.
            mode (str, optional): Mode. `"single"` uses the same LoKr for all roll-out steps,
                and `"all"` uses a different LoKr for every roll-out step. Defaults to `"single"`.
        """
        super().__init__()

        self.mode = mode
        self.max_steps = max_steps
        lokr_layers = max_steps if mode == "all" else 1
        self.lokrs = nn.ModuleList(
            [
                LoKr(in_features, out_features, r=r, alpha=alpha, dropout=dropout)
                for _ in range(lokr_layers)
            ]
        )

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Compute the LoKr adaptation.

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
            return self.lokrs[0](x)
        elif self.mode == "all":
            return self.lokrs[step](x)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
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