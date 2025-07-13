"""Evidential regression head using a Normal-Inverse-Gamma prior."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import StudentT

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model
from ..data import binning as binning_scheme


class EvidentialHead(nn.Module):
    """Small MLP mapping ``x`` to Student-T parameters."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.features = nn.Sequential(*layers)
        self.out = nn.Linear(last, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raw = self.out(self.features(x))
        mu = raw[:, 0:1]
        v = 1 + F.softplus(raw[:, 1:2])
        alpha = 1 + F.softplus(raw[:, 2:3])
        beta = F.softplus(raw[:, 3:4])
        return mu, v, alpha, beta


@register_model("evidential")
class EvidentialModel(BaseModel):
    """Predict a Student-T distribution with evidential uncertainty."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        hidden: Sequence[int] | None = None,
        hidden_dims: Sequence[int] | None = None,
        *,
        lambda_reg: float = 0.01,
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = hidden_dims if hidden_dims is not None else (128, 128)
        self.head = EvidentialHead(in_dim, hidden)
        self.lambda_reg = lambda_reg
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = binning_scheme.BinningScheme(edges=edges)
        self.binner = binner

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bin_logits(x)

    # ------------------------------------------------------------------
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, v, alpha, beta = self.head(x)
        df = 2 * alpha
        scale = torch.sqrt(beta * (1 + v) / (alpha * v) + 1e-8)
        dist = StudentT(df.squeeze(-1), loc=mu.squeeze(-1), scale=scale.squeeze(-1))
        nll = -dist.log_prob(y.squeeze(-1))
        reg = torch.abs(y - mu).squeeze(-1) * (2 * v + alpha).reciprocal().squeeze(-1)
        return -(nll + self.lambda_reg * reg)

    # ------------------------------------------------------------------
    def bin_logits(self, x: torch.Tensor) -> torch.Tensor:
        mu, v, alpha, beta = self.head(x)
        edges = self.binner.edges.to(x)
        scale = torch.sqrt(beta * (1 + v) / (alpha * v) + 1e-8)
        z = (edges[None, :, None] - mu.unsqueeze(1)) / scale.unsqueeze(1)
        cdf = torch.sigmoid(z)
        probs = cdf[:, 1:] - cdf[:, :-1]
        probs = probs.squeeze(-1)
        eps = torch.finfo(probs.dtype).tiny
        return (probs + eps).log()

    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="evidential",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "hidden_dims": [128, 128],
                "lambda_reg": 0.01,
            },
        )
