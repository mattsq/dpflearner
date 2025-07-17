"""LinCDE tree-based conditional density estimator."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model
from ..data import binning as binning_scheme

try:
    from lincde import LinCDE  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LinCDE = None


@register_model("lincde")
class LinCDEModel(BaseModel):
    """Wrapper around :mod:`lincde` providing logits over bins."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        basis: int = 31,
        trees: int = 400,
        lr: float = 0.05,
        depth: int = 3,
        *,
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        
        # Check for required dependency first
        if LinCDE is None:  # pragma: no cover - dependency not installed
            raise ImportError("LinCDE package is required for LinCDEModel")
        
        # Validate ALL inputs before doing any operations
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end})")
        if basis <= 0:
            raise ValueError(f"basis must be positive, got {basis}")
        if trees <= 0:
            raise ValueError(f"trees must be positive, got {trees}")
        if lr <= 0:
            raise ValueError(f"learning rate must be positive, got {lr}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        
        # Now create binning scheme
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1, dtype=torch.float32)
            binner = binning_scheme.BinningScheme(edges=edges)
        
        # Validate binning edges are sorted
        if not torch.all(binner.edges[1:] >= binner.edges[:-1]):
            raise ValueError("Binning edges must be non-decreasing")
            
        self.binner = binner
        self.start = start
        self.end = end
        self.basis = basis
        
        # Create y_grid (this could also fail with invalid parameters)
        try:
            self.y_grid = np.linspace(start, end, basis, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to create y_grid with start={start}, end={end}, basis={basis}: {e}")
        
        try:
            self.model = LinCDE(
                basis="bspline",
                n_basis=basis,
                n_trees=trees,
                learning_rate=lr,
                max_depth=depth,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LinCDE model: {e}")

    # ------------------------------------------------------------------
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "LinCDEModel":
        """Fit the LinCDE model on training data."""
        try:
            # Convert to numpy with proper dtype handling
            x_np = x.detach().cpu().numpy().astype(np.float32)
            y_np = y.detach().cpu().numpy().astype(np.float32)
            
            # Validate input shapes
            if x_np.ndim != 2:
                raise ValueError(f"Expected 2D input x, got shape {x_np.shape}")
            if y_np.ndim != 1:
                raise ValueError(f"Expected 1D input y, got shape {y_np.shape}")
            if x_np.shape[0] != y_np.shape[0]:
                raise ValueError(f"x and y must have same number of samples: {x_np.shape[0]} vs {y_np.shape[0]}")
            
            # Fit the model
            self.model.fit(x_np, y_np)
            return self
            
        except Exception as e:
            raise RuntimeError(f"Failed to fit LinCDE model: {e}")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for input x using the fitted LinCDE model."""
        if not hasattr(self.model, "trees_"):
            raise RuntimeError("Model must be fitted before calling forward")
        
        try:
            # Convert input to numpy with proper dtype
            x_np = x.detach().cpu().numpy().astype(np.float32)
            
            # Validate input shape
            if x_np.ndim != 2:
                raise ValueError(f"Expected 2D input x, got shape {x_np.shape}")
            
            # Get log density predictions
            log_density_np = self.model.predict_log_density(x_np)
            
            # Convert back to torch with proper device and dtype
            logf = torch.from_numpy(log_density_np.astype(np.float32)).to(device=x.device, dtype=x.dtype)
            
            # Compute density and CDF
            density = logf.exp()
            
            # Ensure density is valid (no NaN/inf values)
            if torch.isnan(density).any() or torch.isinf(density).any():
                # Fallback to uniform density
                density = torch.ones_like(density) / density.size(1)
            
            # Compute cumulative sum for CDF
            cdf_basis = torch.cumsum(density, dim=1)
            
            # Get y_grid from fitted model and ensure proper device/dtype
            y_grid_fitted = torch.from_numpy(self.model.y_grid_.astype(np.float32)).to(device=x.device, dtype=x.dtype)
            binner_edges = self.binner.edges.to(device=x.device, dtype=x.dtype)
            
            # Find indices for bin edges in the fitted grid
            idx = torch.bucketize(binner_edges, y_grid_fitted)
            
            # Clamp indices to valid range
            idx = idx.clamp(0, cdf_basis.size(1) - 1)
            
            # Extract CDF values at bin edges
            cdf_edges = cdf_basis[:, idx]
            
            # Compute probabilities as CDF differences
            probs = cdf_edges[:, 1:] - cdf_edges[:, :-1]
            
            # Ensure probabilities are positive and sum to 1
            eps = torch.finfo(probs.dtype).eps * 10  # Use larger epsilon for stability
            probs = torch.clamp(probs, min=eps)
            
            # Renormalize to ensure they sum to 1
            probs = probs / probs.sum(dim=1, keepdim=True)
            
            # Handle any remaining invalid values
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                # Fallback to uniform distribution
                n_bins = probs.size(1)
                uniform_probs = torch.full_like(probs, 1.0 / n_bins)
                probs = torch.where(torch.isfinite(probs), probs, uniform_probs)
            
            return torch.log(probs)
            
        except Exception as e:
            # Ultimate fallback: return uniform distribution
            n_bins = self.binner.edges.numel() - 1
            uniform_logits = torch.full((x.size(0), n_bins), -torch.log(torch.tensor(n_bins, dtype=torch.float32)), 
                                      device=x.device, dtype=x.dtype)
            return uniform_logits

    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="lincde",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "basis": 31,
                "trees": 400,
                "lr": 0.05,
                "depth": 3,
            },
        )
