"""Model registry and helper utilities."""

from __future__ import annotations

from typing import Callable, Dict, Type

from .base import BaseModel
from ..configs.model import ModelConfig

__all__ = ["register_model", "get_model", "MODEL_REGISTRY", "BaseModel"]


MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    """Decorator to register a :class:`BaseModel` implementation."""

    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(cfg: ModelConfig | str, **kwargs) -> BaseModel:
    """Instantiate a model from ``cfg`` or ``name``."""

    if isinstance(cfg, str):
        name = cfg
        params = kwargs
    else:
        name = cfg.name
        params = {**cfg.as_kwargs(), **kwargs}

    model_cls = MODEL_REGISTRY[name]
    return model_cls(**params)

# ------------------------------------------------------------------
# Import built-in model implementations so they register themselves
from . import mlp  # noqa: F401
from . import logistic_regression  # noqa: F401
from . import gaussian_ls  # noqa: F401
from . import mdn  # noqa: F401
from . import quantile_rf  # noqa: F401
from . import ckde  # noqa: F401
from . import lincde  # noqa: F401
from . import rfcde  # noqa: F401
from . import logistic_mixture  # noqa: F401
from . import ngboost_model  # noqa: F401
from . import evidential  # noqa: F401
from . import flow_cde  # noqa: F401
from . import diffusion_cde  # noqa: F401
from . import iqn_model  # noqa: F401
from . import monotone_cdf_model  # noqa: F401
from . import kmn_model  # noqa: F401

from . import imm_jump  # noqa: F401
from . import mean_flow  # noqa: F401
from . import transformer  # noqa: F401
from . import shortcut_cde_model  # noqa: F401
