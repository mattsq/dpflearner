"""Logistic mixture model integrating over bin edges."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from ..utils import make_mlp
from . import register_model
from ..data import binning as binning_scheme


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
        *,
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        
        # Validate inputs
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end})")
        if n_components <= 0:
            raise ValueError(f"n_components must be positive, got {n_components}")
        
        self.n_components = n_components
        out_dim = n_components * 3  # logits, means, log_scales
        self.net = make_mlp(in_dim, out_dim, hidden_dims)
        
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1, dtype=torch.float32)
            binner = binning_scheme.BinningScheme(edges=edges)
        
        # Validate binning edges are sorted
        if not torch.all(binner.edges[1:] >= binner.edges[:-1]):
            raise ValueError("Binning edges must be non-decreasing")
            
        self.binner = binner
        
        # Store constants for numerical stability
        self.min_scale = 1e-6  # Minimum allowed scale
        self.max_scale = 1e3   # Maximum allowed scale

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits by integrating mixture of logistic distributions over bins."""
        # Get mixture parameters from network
        params = self.net(x)  # [batch_size, n_components * 3]
        n = self.n_components
        
        # Split parameters: logits for mixture weights, means, log scales
        logits, means, log_scales = params.split(n, dim=-1)
        
        # Compute mixture weights (normalized)
        weights = torch.softmax(logits, dim=-1)  # [batch_size, n_components]
        
        # Convert log scales to scales with numerical stability bounds
        scales = torch.clamp(log_scales.exp(), min=self.min_scale, max=self.max_scale)
        
        # Handle potential NaN/inf values
        if torch.isnan(means).any() or torch.isinf(means).any():
            means = torch.nan_to_num(means, nan=0.0, posinf=1e3, neginf=-1e3)
        if torch.isnan(scales).any() or torch.isinf(scales).any():
            scales = torch.nan_to_num(scales, nan=1.0, posinf=self.max_scale, neginf=self.min_scale)
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            # Fallback to uniform weights
            weights = torch.full_like(weights, 1.0 / self.n_components)

        # Ensure edges are on correct device and dtype
        edges = self.binner.edges.to(device=x.device, dtype=x.dtype)
        
        # Compute standardized values for CDF computation
        # z = (x - mean) / scale for logistic CDF
        z = (edges.unsqueeze(0).unsqueeze(-1) - means.unsqueeze(1)) / scales.unsqueeze(1)
        # Shape: [batch_size, n_edges, n_components]
        
        # Compute CDF at each bin edge using sigmoid (logistic CDF)
        cdf = torch.sigmoid(z)
        
        # Ensure CDF is properly bounded and monotonic
        cdf = torch.clamp(cdf, min=0.0, max=1.0)
        
        # Compute probabilities for each component as CDF differences
        probs_comp = cdf[..., 1:, :] - cdf[..., :-1, :]  # [batch_size, n_bins, n_components]
        
        # Handle edge case where probabilities might be negative due to numerical errors
        probs_comp = torch.clamp(probs_comp, min=0.0)
        
        # Combine components using mixture weights
        probs = (probs_comp * weights.unsqueeze(1)).sum(dim=-1)  # [batch_size, n_bins]
        
        # Add small epsilon to prevent log(0) and ensure numerical stability
        eps = torch.finfo(probs.dtype).eps * 10  # Use larger epsilon for stability
        probs = probs + eps
        
        # Renormalize probabilities to ensure they sum to 1
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Handle any remaining invalid values
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            # Fallback to uniform distribution
            n_bins = probs.size(-1)
            uniform_probs = torch.full_like(probs, 1.0 / n_bins)
            probs = torch.where(torch.isfinite(probs), probs, uniform_probs)
        
        # Return log probabilities
        return torch.log(probs)

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
