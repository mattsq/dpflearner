"""Loss functions such as cross-entropy or CRPS."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.special import digamma

__all__ = ["cross_entropy", "evidential_loss"]


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return cross-entropy loss for ``logits`` and ``targets``.

    ``torch.nn.functional.cross_entropy`` expects integer class labels, so
    targets are converted to ``long`` if required.  This allows passing
    floating point tensors representing discrete indices.
    """

    if targets.dtype != torch.long:
        targets = targets.long()
    return F.cross_entropy(logits, targets)


def evidential_loss(
    alpha: torch.Tensor, targets: torch.Tensor, lam: float = 1.0
) -> torch.Tensor:
    """Loss for evidential classification with Dirichlet outputs."""

    n_bins = alpha.size(-1)
    t = F.one_hot(targets, num_classes=n_bins).to(dtype=alpha.dtype)
    S = alpha.sum(-1, keepdim=True)
    el = (t * (digamma(S) - digamma(alpha))).sum(-1)
    kl = torch.lgamma(alpha.sum(-1)) - torch.lgamma(torch.tensor(n_bins, dtype=alpha.dtype))
    kl -= (torch.lgamma(alpha) - torch.lgamma(torch.ones_like(alpha))).sum(-1)
    return (el + lam * kl).mean()
