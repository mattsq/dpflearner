"""Common evaluation metrics.

This module exposes lightweight metric functions operating on model
outputs.  Each metric accepts a tensor of logits ``(B, K)`` and the
corresponding integer targets ``(B,)``.  Metrics return scalar tensors
that can be easily converted to Python ``float`` for logging.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch.nn import functional as F

__all__ = ["nll", "accuracy", "METRICS_REGISTRY"]


def nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood computed via cross-entropy."""

    if targets.dtype != torch.long:
        targets = targets.long()
    return F.cross_entropy(logits, targets)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Top-1 accuracy of ``logits`` against ``targets``."""

    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()


METRICS_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "nll": nll,
    "accuracy": accuracy,
}

