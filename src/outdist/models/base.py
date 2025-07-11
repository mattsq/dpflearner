"""Abstract base model class used throughout the project."""

from __future__ import annotations

from typing import Dict, Type

import torch
from torch import nn

from ..configs import model as model_cfg


class Registrable(type):
    """Metaclass that keeps a registry of subclasses."""

    _registry: Dict[str, Type[nn.Module]] = {}

    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict, **kwargs: object) -> Type[nn.Module]:
        cls = super().__new__(mcls, name, bases, namespace)
        # ``_abstract`` marks classes that shouldn't be automatically registered
        if not namespace.get("_abstract", False):
            mcls._registry[name] = cls
        return cls

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """Retrieve a model class by name."""

        return cls._registry[name]

    @classmethod
    def registered(cls) -> Dict[str, Type[nn.Module]]:
        """Return a copy of the registry."""

        return dict(cls._registry)


class BaseModel(nn.Module, metaclass=Registrable):
    """Abstract neural network mapping ``x`` to logits over ``K`` bins."""

    _abstract = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for each output bin given ``x``."""

        raise NotImplementedError

    @classmethod
    def default_config(cls) -> model_cfg.ModelConfig:
        """Return the default configuration for the model."""

        raise NotImplementedError

