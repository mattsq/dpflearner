from __future__ import annotations

from dataclasses import dataclass

from .base import BaseConfig

__all__ = ["TrainerConfig"]

@dataclass
class TrainerConfig(BaseConfig):
    """Configuration options for :class:`~outdist.training.trainer.Trainer`."""

    batch_size: int = 32
    max_epochs: int = 1
    lr: float = 1e-3
    device: str = "cpu"
