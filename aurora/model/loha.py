"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import math
from typing import Literal

import torch
from torch import nn

__all__ = ["LoHa", "LoHaRollout", "LoHaMode"]

LoHaMode = Literal["single", "all"]

class LoHa(nn.Module):
    """LoHa adaptation for a linear layer."""

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
        self.loha_dropout = nn.Dropout(dropout)

        # loha parameters
        self.loha_w1_a = nn.Parameter(torch.empty(r, in_features))
        self.loha_w1_b = nn.Parameter(torch.empty(out_features, r))
        self.loha_w2_a = nn.Parameter(torch.empty(r, in_features))
        self.loha_w2_b = nn.Parameter(torch.empty(out_features, r))
        self.loha_alpha = alpha
        self.scaling = self.loha_alpha / self.r

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        # Initialise A the same way as the default for `nn.Linear` and set B to zero.
        nn.init.kaiming_uniform_(self.loha_w1_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.loha_w1_b, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.loha_w2_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.loha_w2_b, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoHa adaptation.

        Args:
            x (torch.Tensor): Input to the linear layer.

        Returns:
            torch.Tensor: Additive correction for the output of the linear layer.
        """
        # print(x.shape)
        # print(((self.loha_w1_a @ self.loha_w1_b) * (self.loha_w2_a @ self.loha_w2_b)).shape)
        x=self.loha_dropout(x) @ ((self.loha_w1_a.transpose(0,1) @ self.loha_w1_b.transpose(0,1)) * (self.loha_w2_a.transpose(0,1) @ self.loha_w2_b.transpose(0,1)))
        return x * self.scaling


class LoHaRollout(nn.Module):
    """Per-roll-out-step LoHa finetuning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        max_steps: int = 40,
        mode: LoHaMode = "single",
    ):
        """Initialise.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            r (int, optional): Rank. Defaults to `4`.
            alpha (int, optional): Alpha. Defaults to `1`.
            dropout (float, optional): Drop-out rate. Defaults to `0.0`.
            max_steps (int, optional): Maximum number of roll-out steps. Defaults to `40`.
            mode (str, optional): Mode. `"single"` uses the same LoHa for all roll-out steps,
                and `"all"` uses a different LoHa for every roll-out step. Defaults to `"single"`.
        """
        super().__init__()

        self.mode = mode
        self.max_steps = max_steps
        loha_layers = max_steps if mode == "all" else 1
        self.lohas = nn.ModuleList(
            [
                LoHa(in_features, out_features, r=r, alpha=alpha, dropout=dropout)
                for _ in range(loha_layers)
            ]
        )

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Compute the LoHa adaptation.

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
            return self.lohas[0](x)
        elif self.mode == "all":
            return self.lohas[step](x)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
