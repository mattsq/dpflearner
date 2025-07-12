# SPDX-License-Identifier: MIT
# ------------------------------------------
from abc import ABC, abstractmethod
import torch

class BaseConformal(ABC):
    """Set-valued conformal predictor wrapping a probabilistic base model."""

    def __init__(self, base):
        self.base = base

    @abstractmethod
    def calibrate(self, X_cal, y_cal, alpha: float):
        """Fit the conformal adapter on calibration data."""
        ...

    @abstractmethod
    @torch.no_grad()
    def contains(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask indicating set membership."""
        ...

    # ------------------------------------------------------------------
    def coverage(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Return empirical coverage on ``(x, y)``."""
        return self.contains(x, y).float().mean().item()
