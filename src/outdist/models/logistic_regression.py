"""Logistic regression baseline model."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model
from ..data import binning as binning_scheme


@register_model("logreg")
class LogisticRegression(BaseModel):
    """Linear model mapping ``x`` to logits over discretised outcomes."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        *,
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, n_bins)
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = binning_scheme.BinningScheme(edges=edges)
        self.binner = binner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="logreg",
            params={"in_dim": 1, "start": 0.0, "end": 1.0, "n_bins": 10},
        )
