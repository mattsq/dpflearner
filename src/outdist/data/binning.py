"""Binning schemes for discretising continuous targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

__all__ = [
    "BinningScheme",
    "EqualWidthBinning",
    "QuantileBinning",
    "BootstrappedUniformBinning",
    "BootstrappedQuantileBinning",
    "KMeansBinning",
    "BootstrappedKMeansBinning",
    "LearnableBinningScheme",
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


def _kmeans_edges(data: torch.Tensor, n_bins: int, *, random_state: int | None = None) -> torch.Tensor:
    """Return k-means edges for ``data`` with ``n_bins`` clusters."""

    from sklearn.cluster import KMeans

    data_np = data.detach().cpu().numpy().reshape(-1, 1)
    km = KMeans(n_clusters=n_bins, random_state=random_state, n_init="auto")
    km.fit(data_np)
    centers = torch.tensor(km.cluster_centers_.flatten(), device=data.device, dtype=data.dtype)
    centers, _ = torch.sort(centers)
    mids = 0.5 * (centers[1:] + centers[:-1])
    return torch.cat([data.min().unsqueeze(0), mids, data.max().unsqueeze(0)])


class KMeansBinning(BinningScheme):
    """Bins based on 1-D k-means clusters of ``data``."""

    def __init__(self, data: torch.Tensor, n_bins: int, *, random_state: int | None = None) -> None:
        edges = _kmeans_edges(data.flatten(), n_bins, random_state=random_state)
        super().__init__(edges=edges)


class BootstrappedKMeansBinning(BinningScheme):
    """Average k-means bins over bootstrap resamples of ``data``."""

    def __init__(self, data: torch.Tensor, n_bins: int, n_bootstrap: int = 10, *, random_state: int | None = None) -> None:
        data = data.flatten()
        n = data.numel()
        edges_list = []
        for _ in range(n_bootstrap):
            idx = torch.randint(0, n, (n,), device=data.device)
            sample = data[idx]
            edges = _kmeans_edges(sample, n_bins, random_state=random_state)
            edges_list.append(edges)
        mean_edges = torch.stack(edges_list).mean(dim=0)
        super().__init__(edges=mean_edges)


class LearnableBinningScheme(BinningScheme, nn.Module):
    """Binning scheme with trainable cut-points."""

    __hash__ = nn.Module.__hash__

    def __init__(self, n_bins: int, y_min: float, y_max: float, init: str = "uniform") -> None:
        nn.Module.__init__(self)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        if init != "uniform":
            raise ValueError(f"Unknown init '{init}'")
        self.logits = nn.Parameter(torch.zeros(n_bins - 1))

    # ------------------------------------------------------------------
    @property
    def edges(self) -> torch.Tensor:  # type: ignore[override]
        widths = torch.softmax(self.logits, dim=0)
        cumsum = torch.cumsum(widths, dim=0)
        e_inner = self.y_min + (self.y_max - self.y_min) * cumsum
        device = self.logits.device
        dtype = self.logits.dtype
        return torch.cat(
            [
                torch.tensor([self.y_min], device=device, dtype=dtype),
                e_inner,
                torch.tensor([self.y_max], device=device, dtype=dtype),
            ]
        )

    def fit(self, data: torch.Tensor) -> "LearnableBinningScheme":  # type: ignore[override]
        return self
