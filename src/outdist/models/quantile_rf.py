"""Quantile Regression Forest baseline model."""

from __future__ import annotations
import numpy as np

import torch
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model


@register_model("quantile_rf")
class QuantileRandomForest(BaseModel):
    """Approximate conditional density using a random forest."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        n_estimators: int = 100,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        edges = torch.linspace(start, end, n_bins + 1)
        self.register_buffer("bin_edges", edges)

    # ------------------------------------------------------------------
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "QuantileRandomForest":
        """Fit the underlying random forest regressor."""

        self.regressor.fit(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        return self

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits over bins for input ``x``."""

        if not hasattr(self.regressor, "estimators_"):
            raise RuntimeError("Model must be fitted before calling forward")

        X = x.detach().cpu().numpy()
        # Predictions from each tree shape ``(n_samples, n_estimators)``
        tree_preds = np.stack([t.predict(X) for t in self.regressor.estimators_], axis=1)
        edges = self.bin_edges.detach().cpu().numpy()
        n_bins = edges.size - 1
        idx = np.digitize(tree_preds, edges[1:], right=True)
        idx = np.clip(idx, 0, n_bins - 1)
        counts = np.array([
            np.bincount(row, minlength=n_bins) for row in idx
        ])
        probs = counts / counts.sum(axis=1, keepdims=True)
        eps = np.finfo(probs.dtype).tiny
        logits = np.log(np.clip(probs, eps, None))
        return torch.from_numpy(logits).to(x.device)

    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="quantile_rf",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "n_estimators": 100,
                "random_state": None,
            },
        )

