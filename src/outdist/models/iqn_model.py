"""Implicit Quantile Network model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.functional import jacobian

from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model
from ..data import binning as binning_scheme


class QuantileMLP(nn.Module):
    """MLP mapping ``(x, tau)`` to a quantile level."""

    def __init__(self, x_dim: int, hidden: int = 128, K_fourier: int = 16, layers: int = 3) -> None:
        super().__init__()
        self.K = K_fourier
        in_dim = x_dim + 2 * K_fourier
        modules = []
        for i in range(layers):
            modules.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*modules)

    def fourier(self, tau: torch.Tensor) -> torch.Tensor:
        k = torch.arange(1, self.K + 1, device=tau.device, dtype=tau.dtype)
        tau_proj = tau.view(-1, 1) * k.view(1, -1)
        return torch.cat(
            [torch.sin(torch.pi * tau_proj), torch.cos(torch.pi * tau_proj)],
            dim=-1,
        )

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        phi = self.fourier(tau)
        h = torch.cat([x, phi], dim=-1)
        return self.net(h)


@register_model("iqn")
class IQNModel(BaseModel):
    """Implicit Quantile Network producing monotone CDFs."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        *,
        hidden: int = 128,
        K_fourier: int = 16,
        layers: int = 3,
    ) -> None:
        super().__init__()
        self.qnet = QuantileMLP(in_dim, hidden=hidden, K_fourier=K_fourier, layers=layers)
        edges = torch.linspace(start, end, n_bins + 1)
        self.binner = binning_scheme.BinningScheme(edges=edges)
        self.register_buffer("y_min", torch.tensor(start))
        self.register_buffer("y_max", torch.tensor(end))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bin_logits(x)

    # ------------------------------------------------------------------
    def quantile_loss(self, x: torch.Tensor, y: torch.Tensor, *, lambda_reg: float = 1e-3) -> torch.Tensor:
        B = x.size(0)
        tau = torch.rand(B, 1, device=x.device)
        tau = tau.clamp(1e-4, 1 - 1e-4)
        q_hat = self.qnet(x, tau)
        error = y - q_hat
        pinball = (tau - (error < 0).float()) * error
        tau.requires_grad_(True)
        q = self.qnet(x, tau)
        dq_dtau = jacobian(lambda t: self.qnet(x, t), tau).squeeze(-1)
        reg = (dq_dtau - dq_dtau.mean()).abs()
        return pinball.mean() + lambda_reg * reg.mean()

    # ------------------------------------------------------------------
    def quantile(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.clamp(1e-4, 1 - 1e-4)
        return self.qnet(x, tau)

    def cdf(self, x: torch.Tensor, y: torch.Tensor, *, iters: int = 20) -> torch.Tensor:
        lo = torch.zeros_like(y)
        hi = torch.ones_like(y)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            q_mid = self.qnet(x, mid.unsqueeze(1)).squeeze(-1)
            lo = torch.where(q_mid < y, mid, lo)
            hi = torch.where(q_mid >= y, mid, hi)
        return 0.5 * (lo + hi)

    @torch.no_grad()
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tau = self.cdf(x, y).unsqueeze(1)
        tau.requires_grad_(True)
        q = self.qnet(x, tau)
        dq_dtau = jacobian(lambda t: self.qnet(x, t), tau).squeeze(-1)
        return -(dq_dtau + 1e-12).log()

    # ------------------------------------------------------------------
    def bin_logits(self, x: torch.Tensor) -> torch.Tensor:
        edges = self.binner.edges.to(x)
        B = x.size(0)
        edges_rep = edges.repeat(B)
        x_rep = x.repeat_interleave(edges.numel(), dim=0)
        cdf_vals = self.cdf(x_rep, edges_rep).view(B, -1)
        probs = cdf_vals[:, 1:] - cdf_vals[:, :-1]
        eps = torch.finfo(probs.dtype).tiny
        return (probs + eps).log()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, x: torch.Tensor, n: int = 100) -> torch.Tensor:
        B = x.size(0)
        tau = torch.rand(n * B, 1, device=x.device)
        samples = self.qnet(x.repeat_interleave(n, dim=0), tau)
        return samples.view(n, B).T

    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="iqn",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "hidden": 128,
                "K_fourier": 16,
                "layers": 3,
            },
        )

