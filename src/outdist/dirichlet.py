"""Utilities for working with Dirichlet distributions."""

from __future__ import annotations

import torch

__all__ = ["total_uncertainty", "epistemic_entropy"]


def total_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    """Return 1/S-based uncertainty measure (lower is more confident)."""

    K = alpha.size(-1)
    S = alpha.sum(-1)
    return K / (S + 1)


def epistemic_entropy(alpha: torch.Tensor) -> torch.Tensor:
    """Return entropy of the expected categorical distribution."""

    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    return (-p * p.log()).sum(-1)
