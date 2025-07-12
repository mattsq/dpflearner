"""Outdist package: discrete distributions over continuous outcomes."""

from __future__ import annotations

from .models import get_model, register_model  # re-exported for convenience
from .training.trainer import Trainer
from .training.ensemble_trainer import EnsembleTrainer
from .ensembles.average import AverageEnsemble
from .ensembles.stacked import StackedEnsemble
from .data.datasets import make_dataset
from .calibration import get_calibrator, register_calibrator

__all__ = [
    "__version__",
    "get_model",
    "register_model",
    "get_calibrator",
    "register_calibrator",
    "Trainer",
    "EnsembleTrainer",
    "AverageEnsemble",
    "StackedEnsemble",
    "make_dataset",
]

__version__ = "0.1.0"

# Additional lazy imports could be added here when submodules are implemented.
