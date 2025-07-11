"""Loss functions such as cross-entropy or CRPS."""

from __future__ import annotations

import torch
from torch.nn import functional as F

__all__ = ["cross_entropy"]


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return cross-entropy loss for ``logits`` and ``targets``."""

    return F.cross_entropy(logits, targets)
