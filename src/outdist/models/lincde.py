"""LinCDE tree-based conditional density estimator."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model

try:
    from lincde import LinCDE  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LinCDE = None


@register_model("lincde")
class LinCDEModel(BaseModel):
    """Wrapper around :mod:`lincde` providing logits over bins."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        basis: int = 31,
        trees: int = 400,
        lr: float = 0.05,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if LinCDE is None:  # pragma: no cover - dependency not installed
            raise ImportError("LinCDE package is required for LinCDEModel")
        edges = torch.linspace(start, end, n_bins + 1)
        self.register_buffer("bin_edges", edges)
        self.y_grid = np.linspace(start, end, basis)
        self.model = LinCDE(
            basis="bspline",
            n_basis=basis,
            n_trees=trees,
            learning_rate=lr,
            max_depth=depth,
        )

    # ------------------------------------------------------------------
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "LinCDEModel":
        self.model.fit(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        return self

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "trees_"):
            raise RuntimeError("Model must be fitted before calling forward")
        logf = torch.from_numpy(
            self.model.predict_log_density(x.detach().cpu().numpy())
        ).to(x.device)
        density = logf.exp()
        cdf_basis = torch.cumsum(density, dim=1)
        idx = torch.bucketize(
            self.bin_edges.detach().cpu(),
            torch.from_numpy(self.model.y_grid_),
        )
        cdf_edges = cdf_basis[:, idx]
        probs = cdf_edges[:, 1:] - cdf_edges[:, :-1]
        eps = torch.finfo(probs.dtype).tiny
        return torch.log(probs.clamp_min(eps))

    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="lincde",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "basis": 31,
                "trees": 400,
                "lr": 0.05,
                "depth": 3,
            },
        )
