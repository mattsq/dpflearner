"""RFCDE random-forest conditional density estimator."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model
from ..data import binning as binning_scheme

try:
    from rfcde import RFCDE  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    RFCDE = None


@register_model("rfcde")
class RFCDEModel(BaseModel):
    """Wrapper around :mod:`rfcde` returning logits over bins."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        bandwidth: float = 0.2,
        trees: int = 500,
        kde_basis: int = 31,
        min_leaf: int = 5,
        *,
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        if RFCDE is None:  # pragma: no cover - dependency not installed
            raise ImportError("RFCDE package is required for RFCDEModel")
        self.bandwidth = bandwidth
        self.model = RFCDE(
            n_trees=trees,
            mtry=in_dim,
            node_size=min_leaf,
            n_basis=kde_basis,
        )
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = binning_scheme.BinningScheme(edges=edges)
        self.binner = binner

    # ------------------------------------------------------------------
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "RFCDEModel":
        self.model.train(x.detach().cpu().double().numpy(), y.detach().cpu().double().numpy())
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.model, "z_train", None) is None:
            raise RuntimeError("Model must be fitted before calling forward")
        x_np = x.detach().cpu().double().numpy()
        grid = self.binner.edges.cpu().double().numpy()
        pdf = self.model.predict(x_np, grid, bandwidth=self.bandwidth)
        diff = np.diff(grid)
        cdf = np.concatenate([np.zeros((pdf.shape[0], 1)), np.cumsum(0.5 * (pdf[:, :-1] + pdf[:, 1:]) * diff, axis=1)], axis=1)
        probs = torch.from_numpy(cdf[:, 1:] - cdf[:, :-1]).to(x.device)
        eps = torch.finfo(probs.dtype).tiny
        return torch.log(probs.clamp_min(eps))

    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="rfcde",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "bandwidth": 0.2,
                "trees": 500,
                "kde_basis": 31,
                "min_leaf": 5,
            },
        )
