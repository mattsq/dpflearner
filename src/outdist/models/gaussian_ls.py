"""Gaussian location-scale baseline model."""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model
from ..data import binning as binning_scheme


@register_model("gaussian_ls")
class GaussianLocationScale(BaseModel):
    """Predict a Gaussian distribution then integrate over bin edges."""

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
        self.mean_head = nn.Linear(in_dim, 1)
        self.log_std_head = nn.Linear(in_dim, 1)
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = binning_scheme.BinningScheme(edges=edges)
        self.binner = binner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mean_head(x).squeeze(-1)
        log_std = self.log_std_head(x).squeeze(-1)
        std = log_std.exp()
        edges = self.binner.edges.to(x)
        # compute CDF at each bin edge
        z = (edges.unsqueeze(0) - mu.unsqueeze(1)) / (std.unsqueeze(1) * math.sqrt(2))
        cdf = 0.5 * (1 + torch.erf(z))
        probs = cdf[..., 1:] - cdf[..., :-1]
        eps = torch.finfo(probs.dtype).tiny
        logits = torch.log(probs.clamp_min(eps))
        return logits

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="gaussian_ls",
            params={"in_dim": 1, "start": 0.0, "end": 1.0, "n_bins": 10},
        )
