"""Transformer encoder with spline-autoregressive head."""

from __future__ import annotations

import torch
from torch import nn

from .transformer import PositionalEncoding, TransformerBlock
from .base import BaseModel
from . import register_model
from ..configs.model import ModelConfig
from ..heads.rqspline_head import RQSplineHead


@register_model("spline_transformer")
class SplineTransformer(BaseModel):
    """Transformer encoder followed by a spline head."""

    def __init__(
        self,
        in_dim: int = 1,
        n_bins: int = 10,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        n_knots: int = 8,
        dropout: float = 0.1,
        pooling: str = "mean",
        **_: object,
    ) -> None:
        super().__init__()
        # ----- encoder -----------------------------------------------
        self.input_embedding = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, 4 * d_model, dropout) for _ in range(n_layers)]
        )
        self.pooling = pooling
        # ----- spline head -------------------------------------------
        self.head = RQSplineHead(d_model, n_bins, n_knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding(x.unsqueeze(-1))
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        for blk in self.blocks:
            x = blk(x)
        h = {
            "mean": x.mean(1),
            "max": x.max(1)[0],
            "sum": x.sum(1),
            "last": x[:, -1],
        }[self.pooling]
        return self.head(h)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="spline_transformer",
            params=dict(
                in_dim=1,
                n_bins=10,
                d_model=64,
                n_heads=8,
                n_layers=2,
                n_knots=8,
                dropout=0.1,
                pooling="mean",
            ),
        )
