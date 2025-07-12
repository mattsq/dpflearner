from __future__ import annotations

import torch
import torch.nn as nn
from torch.special import digamma

from . import register_calibrator, BaseCalibrator

__all__ = ["DirichletCalibrator"]


@register_calibrator("dirichlet")
class DirichletCalibrator(BaseCalibrator):
    """Post-hoc Dirichlet calibrator from Kull et al. 2019."""

    def __init__(self, n_bins: int, l2: float = 1e-3) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.eye(n_bins))
        self.b = nn.Parameter(torch.zeros(n_bins))
        self.l2 = l2
        self.n_bins = n_bins

    # -------- core mapping --------
    def _alpha(self, probs: torch.Tensor) -> torch.Tensor:
        log_p = torch.log(probs.clamp_min(1e-12))
        v = log_p - log_p.mean(dim=1, keepdim=True)
        z = (v @ self.W.T) + self.b
        return torch.exp(z)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha(probs)
        return alpha / alpha.sum(dim=1, keepdim=True)

    # -------- training on a held-out split --------
    def fit(
        self,
        probs: torch.Tensor,
        y: torch.Tensor,
        max_iter: int = 500,
    ) -> "DirichletCalibrator":
        """Fit the calibrator on validation probabilities."""

        y_onehot = torch.zeros_like(probs).scatter_(1, y[:, None], 1)

        opt = torch.optim.LBFGS(self.parameters(), lr=0.1, max_iter=max_iter)

        def closure() -> torch.Tensor:
            opt.zero_grad()
            alpha = self._alpha(probs)
            S = alpha.sum(dim=1, keepdim=True)
            nll = -(y_onehot * (digamma(alpha) - digamma(S))).sum(dim=1).mean()
            reg = self.l2 * (self.W ** 2).mean()
            loss = nll + reg
            loss.backward()
            return loss

        opt.step(closure)
        return self
