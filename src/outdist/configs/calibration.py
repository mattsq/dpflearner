from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .base import BaseConfig

__all__ = ["CalibratorConfig"]


@dataclass
class CalibratorConfig(BaseConfig):
    """Configuration for calibrators."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def as_kwargs(self) -> Dict[str, Any]:
        """Return stored parameters as kwargs for the calibrator constructor."""

        return dict(self.params)
