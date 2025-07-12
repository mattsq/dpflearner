"""Conditional Kernel Density Estimation model."""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model


@register_model("ckde")
class ConditionalKernelDensityEstimator(BaseModel):
    """Non-parametric conditional density estimator using Gaussian kernels."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        x_bandwidth: float = 1.0,
        y_bandwidth: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.x_bandwidth = x_bandwidth
        self.y_bandwidth = y_bandwidth
        edges = torch.linspace(start, end, n_bins + 1)
        self.register_buffer("bin_edges", edges)
        self.register_buffer("x_train", torch.empty(0, in_dim))
        self.register_buffer("y_train", torch.empty(0))

    # ------------------------------------------------------------------
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "ConditionalKernelDensityEstimator":
        """Store training data for kernel density estimation."""

        self.x_train = x.detach().to(self.bin_edges.device)
        self.y_train = y.detach().to(self.bin_edges.device)
        return self

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_train.numel() == 0:
            raise RuntimeError("Model must be fitted before calling forward")

        xq = x.unsqueeze(1)  # (B, 1, D)
        xb = self.x_train.unsqueeze(0)  # (1, N, D)
        diff = (xq - xb) / self.x_bandwidth
        sqdist = diff.pow(2).sum(dim=-1)  # (B, N)
        weights = torch.exp(-0.5 * sqdist)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        edges = self.bin_edges
        y_train = self.y_train.unsqueeze(0).unsqueeze(0)  # (1,1,N)
        z = (edges.unsqueeze(0).unsqueeze(-1) - y_train) / (
            self.y_bandwidth * math.sqrt(2)
        )
        cdf = 0.5 * (1 + torch.erf(z))
        probs_comp = cdf[..., 1:, :] - cdf[..., :-1, :]
        probs = (probs_comp * weights.unsqueeze(1)).sum(dim=-1)
        eps = torch.finfo(probs.dtype).tiny
        logits = torch.log(probs.clamp_min(eps))
        return logits

    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="ckde",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "x_bandwidth": 1.0,
                "y_bandwidth": 1.0,
            },
        )

