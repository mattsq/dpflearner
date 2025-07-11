"""Base configuration objects for the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

__all__ = ["BaseConfig"]


@dataclass
class BaseConfig:
    """Base class for configuration dataclasses."""

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""

        return asdict(self)
