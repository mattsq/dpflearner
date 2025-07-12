"""Score-based conditional density estimation via diffusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint

from .base import BaseModel
from ..configs.model import ModelConfig
from ..data import binning as binning_scheme
from . import register_model


class MLPScore(nn.Module):
    """Simple MLP used to parameterise the score function."""

    def __init__(self, x_dim: int, hidden: int = 128, layers: int = 5) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU())
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(hidden + x_dim + 1, hidden))
        for _ in range(layers - 1):
            self.net.append(nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, 1)

    def forward(self, y: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([y, x, self.time_embed(t[:, None])], dim=1)
        for layer in self.net:
            h = F.silu(layer(h))
        return self.out(h)


@register_model("diffusion")
class DiffusionCDE(BaseModel):
    """Score-based diffusion model producing per-bin logits."""

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        *,
        sigma_min: float = 1e-3,
        sigma_max: float = 10.0,
        hidden: int = 128,
        layers: int = 5,
        mc_bins: int = 256,
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        self.score = MLPScore(in_dim, hidden, layers)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mc_bins = mc_bins
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = binning_scheme.BinningScheme(edges=edges)
        self.binner = binner

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bin_logits(x)

    # ------------------------------------------------------------------
    def dsm_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        t = torch.rand(B, device=x.device)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        noise = torch.randn_like(y)
        y_t = y + sigma.unsqueeze(1) * noise
        score = self.score(y_t, t, x)
        target = -noise / sigma.unsqueeze(1)
        w = sigma ** 2
        return (w * (score - target).pow(2).sum(1)).mean()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        def ode_func(t: torch.Tensor, y_aug: torch.Tensor) -> torch.Tensor:
            y_curr, logp = y_aug[..., :1], y_aug[..., 1:]
            t_b = t.expand(y_curr.size(0))
            y_curr.requires_grad_(True)
            score = self.score(y_curr, t_b, x)
            (div,) = torch.autograd.grad(score.sum(), y_curr, create_graph=True)
            sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t_b
            dydt = -0.5 * (sigma**2).unsqueeze(-1) * score
            dlogp = 0.5 * sigma**2 * div
            return torch.cat([dydt, dlogp.unsqueeze(-1)], dim=1)

        y_aug0 = torch.cat([y, torch.zeros_like(y)], dim=1)
        t_span = torch.tensor([0.0, 1.0], device=x.device)
        y1, logp1 = odeint(ode_func, y_aug0, t_span)
        base = torch.distributions.Normal(0, self.sigma_max)
        logp_base = base.log_prob(y1[-1]).sum(1, keepdim=True)
        return logp_base.squeeze(1) - logp1[-1].squeeze(1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, x: torch.Tensor, n: int = 100) -> torch.Tensor:
        B = x.size(0)
        y = torch.randn(n * B, 1, device=x.device) * self.sigma_max
        t_steps = torch.linspace(1.0, 0.0, 50, device=x.device)
        for t_next, t_curr in zip(t_steps[:-1], t_steps[1:]):
            t = torch.full((n * B,), t_next, device=x.device)
            score = self.score(y, t, x.repeat_interleave(n, 0))
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            y = y + (sigma_t**2)[:, None] * score * (t_curr - t_next)
            y = y + torch.randn_like(y) * sigma_t.sqrt()[:, None] * (
                t_curr - t_next
            ).sqrt()
        return y.view(n, B).T

    # ------------------------------------------------------------------
    @torch.no_grad()
    def bin_logits(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        samples = self.sample(x, self.mc_bins)  # (B, mc_bins)
        edges = self.binner.edges.to(x)
        idx = torch.bucketize(samples, edges) - 1
        idx = idx.clamp_min(0).clamp_max(edges.numel() - 2)
        probs = torch.zeros(B, edges.numel() - 1, device=x.device)
        ones = torch.ones_like(idx, dtype=probs.dtype)
        probs.scatter_add_(1, idx, ones)
        probs = probs / self.mc_bins
        eps = torch.finfo(probs.dtype).tiny
        return (probs + eps).log()

    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="diffusion",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "sigma_min": 1e-3,
                "sigma_max": 10.0,
                "hidden": 128,
                "layers": 5,
                "mc_bins": 256,
            },
        )

