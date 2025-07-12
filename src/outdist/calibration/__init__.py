from __future__ import annotations

from typing import Callable, Dict, Type

import torch
from torch import nn

from ..configs.calibration import CalibratorConfig

__all__ = [
    "register_calibrator",
    "get_calibrator",
    "CALIBRATOR_REGISTRY",
    "BaseCalibrator",
]

CALIBRATOR_REGISTRY: Dict[str, Type["BaseCalibrator"]] = {}


def register_calibrator(name: str) -> Callable[[Type["BaseCalibrator"]], Type["BaseCalibrator"]]:
    """Decorator to register a :class:`BaseCalibrator` implementation."""

    def decorator(cls: Type[BaseCalibrator]) -> Type[BaseCalibrator]:
        CALIBRATOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_calibrator(cfg: CalibratorConfig | str, **kwargs) -> "BaseCalibrator":
    """Instantiate a calibrator from ``cfg`` or ``name``."""

    if isinstance(cfg, str):
        name = cfg
        params = kwargs
    else:
        name = cfg.name
        params = {**cfg.as_kwargs(), **kwargs}
    cls = CALIBRATOR_REGISTRY[name]
    return cls(**params)


class BaseCalibrator(nn.Module):
    """Base class for calibrators mapping probabilities to probabilities."""

    _abstract = True

    def forward(self, probs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract method
        raise NotImplementedError

    def fit(self, probs: torch.Tensor, y: torch.Tensor) -> "BaseCalibrator":  # pragma: no cover - abstract method
        raise NotImplementedError


# Import built-in calibrators so they register themselves
from . import dirichlet  # noqa: F401
