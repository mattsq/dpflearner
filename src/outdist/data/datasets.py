"""Dataset wrappers and utilities for creating data splits."""

from __future__ import annotations

from typing import Callable, Tuple

import random

import torch
from torch.utils.data import Dataset, TensorDataset, random_split

__all__ = ["make_dataset"]


def make_dataset(
    name: str,
    *,
    n_samples: int = 100,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    n_features: int = 1,
    noise: float = 0.1,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return train/val/test splits for the requested dataset.

    Parameters
    ----------
    name:
        Identifier of the dataset to create. ``"dummy"`` yields random
        classification data while ``"synthetic"`` and ``"synthetic-hard"``
        create continuous regression targets of varying difficulty.
    n_samples:
        Total number of samples to generate.
    splits:
        Fractions for train/val/test. Must sum to 1.
    n_features:
        Number of continuous input features for ``"synthetic"``.
    noise:
        Standard deviation of Gaussian noise added to the target.
    """

    if not torch.isclose(torch.tensor(sum(splits)), torch.tensor(1.0)):
        raise ValueError("splits must sum to 1")

    if name == "dummy":
        x = torch.randn(n_samples, 1)
        y = torch.randint(0, 10, (n_samples,))
        dataset = TensorDataset(x, y)
    elif name == "synthetic":
        x = torch.randn(n_samples, n_features)
        funcs: list[Callable[[torch.Tensor], torch.Tensor]] = [
            lambda t: t,
            lambda t: -t,
            torch.sin,
            torch.cos,
            torch.exp,
            lambda t: 1.0 / (t + 1e-2),
        ]
        chosen = [random.choice(funcs) for _ in range(n_features)]
        y = sum(f(x[:, i]) for i, f in enumerate(chosen))
        y = y + noise * torch.randn(n_samples)
        dataset = TensorDataset(x, y)
    elif name == "synthetic-hard":
        x = torch.randn(n_samples, n_features)
        pair_funcs: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = [
            lambda a, b: torch.sin(a * b),
            lambda a, b: torch.cos(a + b),
            lambda a, b: a * b,
            lambda a, b: torch.exp(-(a ** 2 + b ** 2)),
        ]
        y = torch.zeros(n_samples)
        for i in range(0, n_features - 1, 2):
            f = random.choice(pair_funcs)
            y = y + f(x[:, i], x[:, i + 1])
        if n_features % 2 == 1:
            y = y + torch.sin(x[:, -1])
        y = y + torch.prod(torch.tanh(x), dim=1)
        y = y + noise * torch.randn(n_samples)
        dataset = TensorDataset(x, y)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    n_train = int(n_samples * splits[0])
    n_val = int(n_samples * splits[1])
    n_test = n_samples - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    return train_ds, val_ds, test_ds
