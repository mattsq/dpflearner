"""Model configuration definitions.

The project uses lightweight dataclasses to configure models.  Each
model architecture can define its own dedicated dataclass, but all of
them at least store a ``name`` field so the registry can locate the
appropriate implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

__all__ = ["ModelConfig"]


@dataclass
class ModelConfig:
    """Generic configuration used to instantiate models via ``get_model``."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def as_kwargs(self) -> Dict[str, Any]:
        """Return stored parameters as kwargs for the model constructor."""

        return dict(self.params)
