"""Utility helpers used throughout the project."""

from __future__ import annotations

from typing import Callable, Sequence

import torch.nn as nn

__all__ = ["make_mlp"]


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: int | Sequence[int],
    *,
    activation: Callable[[], nn.Module] = nn.ReLU,
    dropout: float | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
) -> nn.Sequential:
    """Return a simple multi-layer perceptron.

    Parameters
    ----------
    in_dim:
        Number of input features.
    out_dim:
        Number of output features.
    hidden_dims:
        Width of each hidden layer. Can be an ``int`` or a sequence of ``int``.
    activation:
        Factory for the activation modules inserted after each hidden layer.
    dropout:
        If given, ``nn.Dropout`` with this probability is inserted after the
        activation of each hidden layer.
    norm_layer:
        Optional callable returning a normalisation layer given the hidden
        dimension. ``None`` disables normalisation.
    """

    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]

    layers: list[nn.Module] = []
    dims = [in_dim, *hidden_dims]
    for in_features, out_features in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_features, out_features))
        if norm_layer is not None:
            layers.append(norm_layer(out_features))
        if activation is not None:
            # ``activation`` may be a class or a callable returning a module
            act = activation() if isinstance(activation, type) else activation()
            layers.append(act)
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)

