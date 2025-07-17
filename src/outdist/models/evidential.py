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
        df = 2 * alpha
        scale = torch.sqrt(beta * (1 + v) / (alpha * v) + 1e-8)
        
        # Compute bin probabilities by evaluating Student-T at bin centers
        # This is a reasonable approximation for narrow bins
        batch_size = x.shape[0]
        n_bins = len(edges) - 1
        
        # Compute bin centers
        bin_centers = (edges[:-1] + edges[1:]) / 2  # (n_bins,)
        bin_width = edges[1] - edges[0]  # Assuming uniform bins
        
        # Expand bin centers for batch computation
        bin_centers = bin_centers.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_bins)
        
        # Create Student-T distribution and compute probabilities at bin centers
        dist = StudentT(df.squeeze(-1), loc=mu.squeeze(-1), scale=scale.squeeze(-1))
        
        # Compute log probabilities at bin centers for each batch element
        bin_probs = torch.zeros(batch_size, n_bins, device=x.device)
        for i in range(batch_size):
            # Extract distribution parameters for this batch element
            single_dist = StudentT(df[i].item(), loc=mu[i].item(), scale=scale[i].item())
            # Compute probabilities at bin centers
            log_probs = single_dist.log_prob(bin_centers[i])
            bin_probs[i] = torch.exp(log_probs) * bin_width
        
        # Normalize to ensure probabilities sum to 1
        bin_probs = bin_probs / bin_probs.sum(dim=1, keepdim=True)
        
        eps = torch.finfo(bin_probs.dtype).tiny
        return (bin_probs + eps).log()

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
