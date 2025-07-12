"""Conditional normalising flow model."""

from __future__ import annotations

import torch
from torch import nn

from nflows.flows.base import Flow
from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import ReversePermutation
from nflows.distributions.normal import StandardNormal

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model


def _make_transform(hidden_dim: int, context_dim: int, num_bins: int = 8) -> MaskedPiecewiseRationalQuadraticAutoregressiveTransform:
    """Return a conditional autoregressive spline transform."""

    return MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
        features=1,
        hidden_features=hidden_dim,
        context_features=context_dim,
        num_bins=num_bins,
        tails="linear",
        num_blocks=2,
        activation=nn.functional.relu,
    )


def build_flow(context_dim: int, *, n_blocks: int = 5, hidden: int = 64, num_bins: int = 8) -> Flow:
    """Assemble a conditional spline flow."""
    transforms = []
    for _ in range(n_blocks):
        transforms.extend(
            [
                _make_transform(hidden, context_dim, num_bins=num_bins),
                BatchNorm(features=1),
                ReversePermutation(features=1),
            ]
        )
    transform = CompositeTransform(transforms)
    return Flow(transform=transform, distribution=StandardNormal(shape=[1]))


@register_model("flow")
class FlowCDE(BaseModel):
    """Conditional flow producing per-bin logits."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = -5.0,
        end: float = 5.0,
        n_bins: int = 10,
        *,
        blocks: int = 5,
        hidden: int = 64,
        spline_bins: int = 8,
    ) -> None:
        super().__init__()
        edges = torch.linspace(start, end, n_bins + 1)
        self.register_buffer("bin_edges", edges)
        self.flow = build_flow(
            in_dim,
            n_blocks=blocks,
            hidden=hidden,
            num_bins=spline_bins,
        )

    # ------------------------------------------------------------------
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return log density ``log p(y | x)``."""
        return self.flow.log_prob(inputs=y.unsqueeze(-1), context=x)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edges = self.bin_edges.to(x)
        n_edges = edges.numel()
        z, _ = self.flow._transform.forward(
            edges.view(-1, 1).expand(x.size(0), -1, 1).reshape(-1, 1),
            context=x.repeat_interleave(n_edges, dim=0),
        )
        cdf = 0.5 * (1 + torch.erf(z / (2 ** 0.5)))
        cdf = cdf.view(x.size(0), n_edges)
        probs = cdf[:, 1:] - cdf[:, :-1]
        eps = torch.finfo(probs.dtype).tiny
        return torch.log(probs.clamp_min(eps))

    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="flow",
            params={
                "in_dim": 1,
                "start": -5.0,
                "end": 5.0,
                "n_bins": 10,
                "blocks": 5,
                "hidden": 64,
                "spline_bins": 8,
            },
        )
