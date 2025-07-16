"""NGBoost wrapper for discrete-probability forecasting."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor

from . import register_model
from ..configs.model import ModelConfig
from ..data import binning as binning_scheme
from .base import BaseModel

try:  # Optional dependency
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import LogScore
except Exception:  # pragma: no cover - optional dependency
    NGBRegressor = None  # type: ignore
    Normal = None  # type: ignore
    LogScore = None  # type: ignore


@register_model("ngboost")
class NGBoostModel(BaseModel):
    """Natural-Gradient Boosting returning per-bin logits."""

    def __init__(
        self,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        *,
        Dist: str = "normal",
        base_max_depth: int = 3,
        n_estimators: int = 800,
        learning_rate: float = 0.03,
        minibatch_frac: float = 1.0,
        verbose: bool = False,
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        if NGBRegressor is None:  # pragma: no cover - dependency not installed
            raise ImportError("ngboost package is required for NGBoostModel")

        dist_map = {"normal": Normal}
        dist_cls = dist_map[Dist]

        self.regressor = NGBRegressor(
            Dist=dist_cls,
            Base=DecisionTreeRegressor(max_depth=base_max_depth),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            natural_gradient=True,
            minibatch_frac=minibatch_frac,
            verbose=verbose,
            Score=LogScore,
        )
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = binning_scheme.BinningScheme(edges=edges)
        self.binner = binner

    # ------------------------------------------------------------------
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "NGBoostModel":
        """Fit the NGBoost ensemble."""
        self.regressor.fit(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        return self

    # ------------------------------------------------------------------
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities for each fixed bin."""
        edges = self.binner.edges.detach().cpu().numpy()
        dist = self.regressor.pred_dist(x.detach().cpu().numpy())
        cdf = np.vstack([d.cdf(edges) for d in dist])
        probs = np.diff(cdf, axis=1) + 1e-12
        logits = np.log(probs)
        return torch.from_numpy(logits).to(x.device)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_logits(x)

    # ------------------------------------------------------------------
    @staticmethod
    def default_config() -> ModelConfig:
        return ModelConfig(
            name="ngboost",
            params={
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "Dist": "normal",
                "base_max_depth": 3,
                "n_estimators": 800,
                "learning_rate": 0.03,
                "minibatch_frac": 1.0,
                "verbose": False,
            },
        )

