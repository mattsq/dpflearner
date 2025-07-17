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

__all__ = ["nll", "accuracy", "crps", "METRICS_REGISTRY"]


def nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood computed via cross-entropy."""

    if targets.dtype != torch.long:
        targets = targets.long()
    return F.cross_entropy(logits, targets)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Top-1 accuracy of ``logits`` against ``targets``."""

    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()


def crps(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Continuous Ranked Probability Score for discrete distributions.
    
    Computes CRPS using the discrete version formula:
    CRPS = sum_k (F_k - I(y <= k))^2
    
    where F_k is the cumulative probability up to bin k,
    and I(y <= k) is the indicator function.
    
    Args:
        logits: Model output logits of shape (B, K)
        targets: Target bin indices of shape (B,)
    
    Returns:
        Scalar tensor containing the mean CRPS across the batch
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # (B, K)
    
    # Compute cumulative distribution function
    cdf = torch.cumsum(probs, dim=-1)  # (B, K)
    
    # Create indicator matrix: I(y <= k) for each target
    batch_size, n_bins = probs.shape
    targets = targets.long()
    
    # Create a matrix where entry (i, k) = 1 if targets[i] <= k, else 0
    bin_indices = torch.arange(n_bins, device=logits.device).unsqueeze(0)  # (1, K)
    target_indices = targets.unsqueeze(1)  # (B, 1)
    indicator = (target_indices <= bin_indices).float()  # (B, K)
    
    # Compute CRPS for each sample: sum_k (F_k - I(y <= k))^2
    squared_diff = (cdf - indicator) ** 2  # (B, K)
    crps_per_sample = squared_diff.sum(dim=-1)  # (B,)
    
    # Return mean CRPS across the batch
    return crps_per_sample.mean()


METRICS_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "nll": nll,
    "accuracy": accuracy,
    "crps": crps,
}

