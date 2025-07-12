"""Evidential neural network architecture."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseModel
from ..configs.model import ModelConfig
from ..utils import make_mlp
from . import register_model


@register_model("evidential")
class EvidentialNet(BaseModel):
    """Predict Dirichlet concentration parameters for categorical outcomes."""

    def __init__(
        self,
        in_dim: int = 1,
        n_bins: int = 10,
        hidden_dims: int | Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        backbone_dims = list(hidden_dims[:-1])
        last_dim = hidden_dims[-1]
        self.backbone = make_mlp(in_dim, last_dim, backbone_dims)
        self.evidence_head = nn.Linear(last_dim, n_bins)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        evidence = F.softplus(self.evidence_head(features))
        alpha = evidence + 1.0
        S = alpha.sum(-1, keepdim=True)
        probs = alpha / S
        logits = probs.log()
        return {"alpha": alpha, "probs": probs, "logits": logits, "S": S}

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="evidential",
            params={"in_dim": 1, "n_bins": 10, "hidden_dims": [128, 128]},
        )
