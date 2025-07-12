"""Logistic mixture model integrating over bin edges."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from ..utils import make_mlp
from . import register_model


@register_model("logistic_mixture")
class LogisticMixture(BaseModel):
    """Predict a mixture of logistic components and integrate over bins."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        n_components: int = 3,
        hidden_dims: int | tuple[int, ...] = (32, 32),
    ) -> None:
        super().__init__()
        self.n_components = n_components
        out_dim = n_components * 3
        self.net = make_mlp(in_dim, out_dim, hidden_dims)
        edges = torch.linspace(start, end, n_bins + 1)
        self.register_buffer("bin_edges", edges)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.net(x)
        n = self.n_components
        logits, means, log_scales = params.split(n, dim=-1)
        weights = torch.softmax(logits, dim=-1)
        scales = log_scales.exp()

        edges = self.bin_edges
        z = (edges.unsqueeze(0).unsqueeze(-1) - means.unsqueeze(1)) / scales.unsqueeze(1)
        cdf = torch.sigmoid(z)
        probs_comp = cdf[..., 1:, :] - cdf[..., :-1, :]
        probs = (probs_comp * weights.unsqueeze(1)).sum(dim=-1)
        eps = torch.finfo(probs.dtype).tiny
        logits = torch.log(probs.clamp_min(eps))
        return logits

    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="logistic_mixture",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "n_components": 3,
                "hidden_dims": [32, 32],
            },
        )
