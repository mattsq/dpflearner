"""Utilities for representing discrete distributions over continuous targets."""

from __future__ import annotations

from typing import TypedDict

import torch


class DistributionPrediction(TypedDict):
    """Prediction object representing a discrete distribution.

    Parameters are stored as a probability vector over ``K`` bins and a
    tensor of bin edges with length ``K+1``.
    """

    probs: torch.Tensor  # shape ``(B, K)``
    bin_edges: torch.Tensor  # shape ``(K+1,)``

    # ------------------------------------------------------------------
    # Moment helpers
    # ------------------------------------------------------------------
    def mean(self) -> torch.Tensor:
        """Return the approximate mean of the distribution."""

        centers = 0.5 * (self["bin_edges"][:-1] + self["bin_edges"][1:])
        return (self["probs"] * centers).sum(dim=-1)

    def var(self) -> torch.Tensor:
        """Return the approximate variance of the distribution."""

        centers = 0.5 * (self["bin_edges"][:-1] + self["bin_edges"][1:])
        mean = self.mean()
        return (self["probs"] * centers.pow(2)).sum(dim=-1) - mean.pow(2)

    # ------------------------------------------------------------------
    # Distributional utilities
    # ------------------------------------------------------------------
    def cdf(self, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the cumulative distribution function at ``y``.

        The input ``y`` should broadcast with the batch dimension of
        ``probs``.  Values below the first bin edge map to ``0`` and
        values beyond the last edge map to ``1``.
        """

        edges = self["bin_edges"]
        probs = self["probs"]
        cum_probs = probs.cumsum(dim=-1)

        # Determine which bin each ``y`` falls into.
        idx = torch.bucketize(y, edges[1:], right=False)
        idx_clamped = idx.clamp(max=probs.shape[-1] - 1)

        before = torch.where(
            idx_clamped == 0,
            torch.zeros_like(y, dtype=probs.dtype),
            cum_probs.gather(-1, idx_clamped - 1),
        )

        lower = edges[idx_clamped]
        upper = edges[idx_clamped + 1]
        width = upper - lower
        frac = ((y - lower) / width).clamp(0.0, 1.0)
        partial = probs.gather(-1, idx_clamped) * frac

        cdf = before + partial
        cdf = torch.where(y < edges[0], torch.zeros_like(cdf), cdf)
        cdf = torch.where(y >= edges[-1], torch.ones_like(cdf), cdf)
        return cdf

    def sample(self, n: int = 1) -> torch.Tensor:
        """Draw ``n`` samples for each batch element."""

        edges = self["bin_edges"]
        probs = self["probs"]

        cat = torch.distributions.Categorical(probs=probs)
        bin_idx = cat.sample((n,))  # ``(n, B)``
        u = torch.rand_like(bin_idx, dtype=edges.dtype)
        lower = edges[bin_idx]
        upper = edges[bin_idx + 1]
        samples = lower + u * (upper - lower)
        return samples.transpose(0, 1)  # ``(B, n)``

