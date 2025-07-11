"""Simple multilayer perceptron model."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from ..utils import make_mlp
from . import register_model


@register_model("mlp")
class MLP(BaseModel):
    """Plain MLP mapping ``x`` to logits over discretised outcomes."""

    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 10,
        hidden_dims: int | Sequence[int] = (32, 32),
    ) -> None:
        super().__init__()
        self.net = make_mlp(in_dim, out_dim, hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="mlp", params={"in_dim": 1, "out_dim": 10, "hidden_dims": [32, 32]}
        )


