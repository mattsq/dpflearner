"""Simple dataset transformation helpers.

This module intentionally keeps the functionality minimal.  The aim is to
provide a couple of basic transforms that are easy to compose in tests and
examples without pulling in heavy dependencies.  Users can of course roll their
own transforms or use libraries like ``torchvision`` in more complex setups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

__all__ = ["Standardise", "AddNoise", "Compose"]


class Standardise:
    """Standardise data to zero mean and unit variance."""

    def __init__(self) -> None:
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def fit(self, data: torch.Tensor) -> "Standardise":
        """Compute ``mean`` and ``std`` for ``data`` along the 0th dimension."""

        self._mean = data.mean(dim=0, keepdim=True)
        self._std = data.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        return self

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Return standardised ``data`` using previously computed statistics."""

        if self._mean is None or self._std is None:
            raise RuntimeError("Standardise must be fitted before calling")
        return (data - self._mean) / self._std


@dataclass
class AddNoise:
    """Add Gaussian noise with standard deviation ``std``."""

    std: float = 1.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(data) * self.std
        return data + noise


class Compose:
    """Compose several transforms in sequence."""

    def __init__(self, transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]]):
        self.transforms = list(transforms)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            data = t(data)
        return data

