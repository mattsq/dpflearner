"""Dataset wrappers and utilities for creating data splits."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset, TensorDataset, random_split

__all__ = ["make_dataset"]


def make_dataset(name: str, *, n_samples: int = 100, splits: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> tuple[Dataset, Dataset, Dataset]:
    """Return train/val/test splits for the requested dataset.

    Parameters
    ----------
    name:
        Identifier of the dataset to create. Currently only ``"dummy"`` is
        supported and yields randomly generated data.
    n_samples:
        Total number of samples to generate. Only used for ``"dummy"``.
    splits:
        Fractions for train/val/test. Must sum to 1.
    """

    if not torch.isclose(torch.tensor(sum(splits)), torch.tensor(1.0)):
        raise ValueError("splits must sum to 1")

    if name == "dummy":
        x = torch.randn(n_samples, 1)
        y = torch.randint(0, 10, (n_samples,))
        dataset = TensorDataset(x, y)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    n_train = int(n_samples * splits[0])
    n_val = int(n_samples * splits[1])
    n_test = n_samples - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    return train_ds, val_ds, test_ds
