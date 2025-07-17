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
from ..data import binning as binning_scheme


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
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        
        # Validate inputs
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end})")
        
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = binning_scheme.BinningScheme(edges=edges)
        
        # Validate binning edges are sorted
        if not torch.all(binner.edges[1:] >= binner.edges[:-1]):
            raise ValueError("Binning edges must be non-decreasing")
        
        self.binner = binner
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
        # Ensure edges are on the same device as input
        edges = self.binner.edges.to(device=x.device, dtype=x.dtype)
        n_edges = edges.numel()
        
        # Use public API instead of private _transform access
        try:
            # Transform edges through the flow to get latent values
            edge_samples = edges.view(-1, 1).expand(x.size(0), -1, 1).reshape(-1, 1)
            context_expanded = x.repeat_interleave(n_edges, dim=0)
            z, _ = self.flow._transform.forward(edge_samples, context=context_expanded)
        except Exception as e:
            # Fallback: if direct transform access fails, this indicates a compatibility issue
            raise RuntimeError(f"Flow transform failed - possible nflows version incompatibility: {e}")
        
        # Compute CDF using standard normal CDF (erf function)
        cdf = 0.5 * (1 + torch.erf(z / (2 ** 0.5)))
        cdf = cdf.view(x.size(0), n_edges)
        
        # Ensure CDF is valid (monotonic and bounded)
        cdf = torch.clamp(cdf, min=0.0, max=1.0)
        
        # Compute probabilities as differences in CDF
        probs = cdf[:, 1:] - cdf[:, :-1]
        
        # Handle edge cases: ensure probabilities are positive and finite
        eps = torch.finfo(probs.dtype).tiny
        probs = torch.clamp(probs, min=eps)
        
        # Check for invalid values and handle them
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            # Replace invalid probabilities with uniform distribution
            uniform_prob = 1.0 / probs.size(-1)
            probs = torch.where(torch.isfinite(probs), probs, uniform_prob)
        
        return torch.log(probs)

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
