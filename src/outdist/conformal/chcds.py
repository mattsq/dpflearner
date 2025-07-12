# SPDX-License-Identifier: MIT
# ------------------------------------------
import torch
import numpy as np

from .base import BaseConformal


class CHCDSConformal(BaseConformal):
    """Conformal Highest Conditional Density Sets."""

    def __init__(self, base, tau: float = 0.90, mode: str = "sub", eps: float = 1e-12) -> None:
        super().__init__(base)
        assert mode in {"sub", "div"}
        self.tau = tau
        self.mode = mode
        self.eps = eps
        self.delta: float | None = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _hd_cutoff(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base.bin_logits(x)
        probs = logits.exp()
        widths = self.base.bins.edges[1:] - self.base.bins.edges[:-1]
        rho = probs / widths

        sort_idx = torch.argsort(rho, dim=1, descending=True)
        rho_s = torch.gather(rho, 1, sort_idx)
        prob_s = torch.gather(probs, 1, sort_idx)
        cum_mass = prob_s.cumsum(dim=1)
        k_star = (cum_mass < self.tau).sum(dim=1).clamp(max=rho.size(1) - 1)
        rows = torch.arange(x.size(0), device=x.device)
        return rho_s[rows, k_star]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, X_cal, y_cal, alpha: float):
        x = torch.as_tensor(X_cal, dtype=torch.float32)
        y = torch.as_tensor(y_cal, dtype=torch.float32)
        c_hd = self._hd_cutoff(x)
        dens = self.base.log_prob(x, y).exp()
        if self.mode == "sub":
            scores = c_hd - dens
        else:
            scores = c_hd / (dens + self.eps)
        m = len(scores)
        q = torch.quantile(
            scores,
            (1 - alpha) * (m + 1) / m,
            interpolation="higher",
        )
        self.delta = q.item()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _cutoff(self, x: torch.Tensor) -> torch.Tensor:
        base = self._hd_cutoff(x)
        if self.mode == "sub":
            return (base - self.delta).clamp(min=0.0)
        return base / max(self.delta, self.eps)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def contains(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dens = self.base.log_prob(x, y).exp()
        return dens >= self._cutoff(x)
