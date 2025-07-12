"""Binning schemes for discretising continuous targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

__all__ = [
    "BinningScheme",
    "EqualWidthBinning",
    "QuantileBinning",
    "BootstrappedUniformBinning",
    "BootstrappedQuantileBinning",
]


@dataclass
class BinningScheme:
    """Map continuous values to discrete bin indices via fixed edges."""

    edges: torch.Tensor

    def __post_init__(self) -> None:
        if self.edges.ndim != 1:
            raise ValueError("edges must be 1D")
        if self.edges.numel() < 2:
            raise ValueError("edges must contain at least two values")
        if not torch.all(self.edges[1:] > self.edges[:-1]):
            raise ValueError("edges must be strictly increasing")
        # Ensure floating dtype for subsequent computations
        self.edges = self.edges.to(dtype=torch.get_default_dtype())

    # ------------------------------------------------------------------
    def fit(self, data: torch.Tensor) -> "BinningScheme":
        """Return ``self`` for API compatibility."""

        return self

    @property
    def n_bins(self) -> int:
        """Number of discrete bins."""

        return self.edges.numel() - 1

    # ------------------------------------------------------------------
    def to_index(self, y: torch.Tensor) -> torch.Tensor:
        """Return integer bin indices for ``y``."""

        idx = torch.bucketize(y, self.edges[1:], right=True)
        return idx.clamp(max=self.n_bins - 1)

    def bin_edges(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(lower, upper)`` edges for bins ``idx``."""

        return self.edges[idx], self.edges[idx + 1]

    def centers(self) -> torch.Tensor:
        """Return bin center locations."""

        return 0.5 * (self.edges[:-1] + self.edges[1:])


class EqualWidthBinning(BinningScheme):
    """Uniform bin widths between ``start`` and ``end``."""

    def __init__(self, start: float, end: float, n_bins: int) -> None:
        edges = torch.linspace(start, end, n_bins + 1)
        super().__init__(edges=edges)


class QuantileBinning(BinningScheme):
    """Bins based on quantiles of ``data``."""

    def __init__(self, data: torch.Tensor, n_bins: int) -> None:
        probs = torch.linspace(0, 1, n_bins + 1, device=data.device)
        edges = torch.quantile(data.flatten(), probs)
        super().__init__(edges=edges)


class BootstrappedUniformBinning(BinningScheme):
    """Average equal-width bins over bootstrap resamples of ``data``."""

    def __init__(self, data: torch.Tensor, n_bins: int, n_bootstrap: int = 10) -> None:
        data = data.flatten()
        n = data.numel()
        edges_list = []
        for _ in range(n_bootstrap):
            idx = torch.randint(0, n, (n,), device=data.device)
            sample = data[idx]
            start = sample.min()
            end = sample.max()
            edges = torch.linspace(start, end, n_bins + 1, device=data.device)
            edges_list.append(edges)
        mean_edges = torch.stack(edges_list).mean(dim=0)
        super().__init__(edges=mean_edges)


class BootstrappedQuantileBinning(BinningScheme):
    """Average quantile bins over bootstrap resamples of ``data``."""

    def __init__(self, data: torch.Tensor, n_bins: int, n_bootstrap: int = 10) -> None:
        data = data.flatten()
        n = data.numel()
        probs = torch.linspace(0, 1, n_bins + 1, device=data.device)
        edges_list = []
        for _ in range(n_bootstrap):
            idx = torch.randint(0, n, (n,), device=data.device)
            sample = data[idx]
            edges = torch.quantile(sample, probs)
            edges_list.append(edges)
        mean_edges = torch.stack(edges_list).mean(dim=0)
        super().__init__(edges=mean_edges)
