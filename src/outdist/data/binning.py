"""Binning schemes for discretising continuous targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import nn

__all__ = [
    "BinningScheme",
    "EqualWidthBinning",
    "QuantileBinning",
    "FreedmanDiaconisBinning",
    "KMeansBinning",
    "LearnableBinningScheme",
    "bootstrap",
]


def bootstrap(
    strategy: Callable[[torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    *,
    n_bootstrap: int = 10,
) -> "BinningScheme":
    """Return averaged bin edges over ``n_bootstrap`` resamples of ``data``.

    Parameters
    ----------
    strategy:
        Callable that maps a bootstrap sample to a 1-D tensor of edges.
    data:
        1-D tensor of target values to resample from.
    n_bootstrap:
        Number of bootstrap draws used to average bin edges.
    """

    data = data.flatten()
    n = data.numel()
    edges_list = []
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), device=data.device)
        sample = data[idx]
        edges = strategy(sample)
        edges_list.append(edges)
    mean_edges = torch.stack(edges_list).mean(dim=0)
    return BinningScheme(edges=mean_edges)


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




class FreedmanDiaconisBinning(BinningScheme):
    """Bins with width from the Freedman--Diaconis rule."""

    def __init__(self, data: torch.Tensor) -> None:
        data = data.flatten()
        n = data.numel()
        q1, q3 = torch.quantile(data, torch.tensor([0.25, 0.75], device=data.device))
        iqr = q3 - q1
        h = 2 * iqr / (n ** (1.0 / 3.0))
        if h <= 0:
            h = torch.finfo(data.dtype).eps
        start = data.min()
        end = data.max()
        edges = torch.arange(start, end + h, h, device=data.device, dtype=data.dtype)
        edges[-1] = end
        super().__init__(edges=edges)




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




class LearnableBinningScheme(BinningScheme, nn.Module):
    """Binning scheme with trainable cut-points.

    Parameters
    ----------
    start:
        Lower bound of the binning range.
    end:
        Upper bound of the binning range.
    n_bins:
        Number of discrete bins.
    init:
        Initialisation strategy for the cut-points. Currently only
        ``"uniform"`` is supported.
    """

    __hash__ = nn.Module.__hash__

    def __init__(
        self, start: float, end: float, n_bins: int, *, init: str = "uniform"
    ) -> None:
        nn.Module.__init__(self)
        self.start = float(start)
        self.end = float(end)
        if init != "uniform":
            raise ValueError(f"Unknown init '{init}'")
        self.logits = nn.Parameter(torch.zeros(n_bins - 1))

    # ------------------------------------------------------------------
    @property
    def edges(self) -> torch.Tensor:  # type: ignore[override]
        widths = torch.softmax(self.logits, dim=0)
        cumsum = torch.cumsum(widths, dim=0)
        e_inner = self.start + (self.end - self.start) * cumsum
        device = self.logits.device
        dtype = self.logits.dtype
        return torch.cat(
            [
                torch.tensor([self.start], device=device, dtype=dtype),
                e_inner,
                torch.tensor([self.end], device=device, dtype=dtype),
            ]
        )

    def fit(self, data: torch.Tensor) -> "LearnableBinningScheme":  # type: ignore[override]
        return self
