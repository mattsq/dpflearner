"""Evaluation utilities for cross-validation and log-likelihood tables.

This module exposes a small :func:`cross_validate` helper used in the tests
to run ``k``-fold cross validation.  It intentionally keeps state and features
to a minimum; the aim is simply to demonstrate the design sketched in
``prompt.md``.  More sophisticated functionality can be layered on later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import torch
from torch.utils.data import Dataset, Subset

from ..configs.trainer import TrainerConfig
from ..models.base import BaseModel
from ..training.trainer import Trainer

__all__ = ["CVFoldResult", "cross_validate"]


@dataclass
class CVFoldResult:
    """Metrics for a single cross-validation fold."""

    fold: int
    metrics: Dict[str, float]


def _split_dataset(dataset: Dataset, k_folds: int) -> Iterable[tuple[Subset, Subset]]:
    """Yield ``(train_ds, val_ds)`` splits for ``k_folds`` cross validation."""

    n = len(dataset)
    fold_size = n // k_folds
    indices = list(range(n))

    for i in range(k_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = indices[:val_start] + indices[val_end:]
        yield Subset(dataset, train_idx), Subset(dataset, val_idx)


def cross_validate(
    model_factory: Callable[[], BaseModel],
    dataset: Dataset,
    *,
    k_folds: int = 5,
    metrics: List[str] | None = None,
    trainer_cfg: TrainerConfig | None = None,
) -> List[CVFoldResult]:
    """Run ``k``-fold cross validation on ``dataset``.

    Parameters
    ----------
    model_factory:
        Callable returning a fresh :class:`BaseModel` instance for each fold.
    dataset:
        The dataset to split into ``k_folds`` parts.
    k_folds:
        Number of folds to run.  Defaults to ``5``.
    metrics:
        Metrics to compute via :class:`Trainer.evaluate`.  Defaults to ``["nll"]``.
    trainer_cfg:
        Optional configuration passed to each :class:`Trainer`.
    """

    if trainer_cfg is None:
        trainer_cfg = TrainerConfig()

    results: List[CVFoldResult] = []
    for fold, (train_ds, val_ds) in enumerate(_split_dataset(dataset, k_folds), start=1):
        trainer = Trainer(trainer_cfg)
        model = model_factory()
        ckpt = trainer.fit(model, train_ds, val_ds)
        metrics_dict = trainer.evaluate(ckpt.model, val_ds, metrics=metrics)
        results.append(CVFoldResult(fold=fold, metrics=metrics_dict))

    return results

