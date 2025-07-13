"""Evidential regression model predicting a Student-T distribution."""

from __future__ import annotations

from typing import Sequence

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import StudentT

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model
from ..data import binning as binning_scheme
from ..utils import make_mlp


class EvidentialHead(nn.Module):
    """MLP mapping ``x`` to Student-T parameters."""

    def __init__(self, in_dim: int, hidden_dims: int | Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.net = make_mlp(in_dim, 4, hidden_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        mu = raw[:, 0:1]
        v = 1.0 + F.softplus(raw[:, 1:2])
        alpha = 1.0 + F.softplus(raw[:, 2:3])
        beta = F.softplus(raw[:, 3:4])
        return mu, v, alpha, beta


@register_model("evidential")
class EvidentialModel(BaseModel):
    """Evidential head producing per-bin logits and log-probabilities."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        *,
        hidden_dims: int | Sequence[int] = (128, 128),
        lambda_reg: float = 0.01,
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        self.head = EvidentialHead(in_dim, hidden_dims)
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
        df = (2 * alpha).squeeze(-1)
        scale = torch.sqrt(beta * (1 + v) / (alpha * v) + 1e-8).squeeze(-1)
        dist = StudentT(df, loc=mu.squeeze(-1), scale=scale)
        nll = -dist.log_prob(y.squeeze(-1))
        reg = torch.abs(y - mu).squeeze(-1) * (2 * v + alpha).squeeze(-1).reciprocal()
        return -(nll + self.lambda_reg * reg)

    # ------------------------------------------------------------------
    def bin_logits(self, x: torch.Tensor) -> torch.Tensor:
        mu, v, alpha, beta = self.head(x)
        df = (2 * alpha).squeeze(-1)
        scale = torch.sqrt(beta * (1 + v) / (alpha * v) + 1e-8).squeeze(-1)
        def student_t_cdf(val: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
            # Approximate using standard normal CDF for differentiability
            return 0.5 * (1 + torch.erf(val / math.sqrt(2)))
        edges = self.binner.edges.to(x)
        z = (edges[None, :] - mu) / scale.unsqueeze(-1)
        cdf = student_t_cdf(z, df.unsqueeze(-1))
        probs = cdf[:, 1:] - cdf[:, :-1]
        eps = torch.finfo(probs.dtype).tiny
        return torch.log(probs.clamp_min(eps))

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
