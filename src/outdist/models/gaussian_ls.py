"""Gaussian location-scale baseline model."""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model
from ..data import binning as binning_scheme


@register_model("gaussian_ls")
class GaussianLocationScale(BaseModel):
    """Predict a Gaussian distribution then integrate over bin edges."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
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
        
        self.mean_head = nn.Linear(in_dim, 1)
        self.log_std_head = nn.Linear(in_dim, 1)
        
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1, dtype=torch.float32)
            binner = binning_scheme.BinningScheme(edges=edges)
        
        # Validate binning edges are sorted
        if not torch.all(binner.edges[1:] >= binner.edges[:-1]):
            raise ValueError("Binning edges must be non-decreasing")
            
        self.binner = binner
        
        # Store constants for numerical stability
        self.min_std = 1e-6  # Minimum allowed standard deviation
        self.max_std = 1e3   # Maximum allowed standard deviation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits by integrating Gaussian distribution over bins."""
        # Get mean and log std predictions
        mu = self.mean_head(x).squeeze(-1)  # [batch_size]
        log_std = self.log_std_head(x).squeeze(-1)  # [batch_size]
        
        # Convert to std with numerical stability bounds
        std = torch.clamp(log_std.exp(), min=self.min_std, max=self.max_std)
        
        # Handle potential NaN/inf values
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            mu = torch.nan_to_num(mu, nan=0.0, posinf=1e3, neginf=-1e3)
        if torch.isnan(std).any() or torch.isinf(std).any():
            std = torch.nan_to_num(std, nan=1.0, posinf=self.max_std, neginf=self.min_std)
        
        # Ensure edges are on correct device and dtype
        edges = self.binner.edges.to(device=x.device, dtype=x.dtype)
        
        # Compute standardized values for CDF computation
        # z = (x - mu) / (std * sqrt(2)) for erf function
        sqrt_2 = torch.tensor(math.sqrt(2), device=x.device, dtype=x.dtype)
        z = (edges.unsqueeze(0) - mu.unsqueeze(1)) / (std.unsqueeze(1) * sqrt_2)
        
        # Compute CDF at each bin edge using erf
        # CDF(x) = 0.5 * (1 + erf((x - mu) / (std * sqrt(2))))
        cdf = 0.5 * (1 + torch.erf(z))
        
        # Ensure CDF is properly bounded and monotonic
        cdf = torch.clamp(cdf, min=0.0, max=1.0)
        
        # Compute probabilities as CDF differences
        probs = cdf[..., 1:] - cdf[..., :-1]
        
        # Handle edge case where probabilities might be negative due to numerical errors
        probs = torch.clamp(probs, min=0.0)
        
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

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="gaussian_ls",
            params={"in_dim": 1, "start": 0.0, "end": 1.0, "n_bins": 10},
        )
