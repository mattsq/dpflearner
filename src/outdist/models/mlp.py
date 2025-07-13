"""Simple multilayer perceptron model."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from ..utils import make_mlp
from . import register_model
from ..data import binning as binning_scheme


@register_model("mlp")
class MLP(BaseModel):
    """Plain MLP mapping ``x`` to logits over discretised outcomes."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        hidden_dims: int | Sequence[int] = (32, 32),
        *,
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(in_dim, n_bins, hidden_dims)
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = binning_scheme.BinningScheme(edges=edges)
        self.binner = binner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="mlp",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "hidden_dims": [32, 32],
            },
        )


