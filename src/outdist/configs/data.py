"""Data loading and preprocessing configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .base import BaseConfig

__all__ = ["DataConfig"]


@dataclass
class DataConfig(BaseConfig):
    """Configuration options for dataset creation."""

    name: str = "dummy"
    n_samples: int = 100
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1)

    def as_kwargs(self) -> Dict[str, object]:
        """Return arguments for :func:`outdist.data.datasets.make_dataset`."""

        return {"n_samples": self.n_samples, "splits": self.splits}
