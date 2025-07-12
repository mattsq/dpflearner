"""Logistic regression baseline model."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model


@register_model("logreg")
class LogisticRegression(BaseModel):
    """Linear model mapping ``x`` to logits over discretised outcomes."""

    def __init__(self, in_dim: int = 1, out_dim: int = 10) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(name="logreg", params={"in_dim": 1, "out_dim": 10})
