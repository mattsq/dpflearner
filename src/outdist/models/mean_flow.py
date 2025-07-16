from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.functional import jvp

from . import register_model
from .base import BaseModel
from ..utils import make_mlp
from ..data.binning import BinningScheme


@register_model("mean_flow")
class MeanFlow(BaseModel):
    """Predict the average velocity from time ``s`` to ``t``."""

    def __init__(
        self,
        in_dim: int = 1,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        time_embed_dim: int = 32,
        sigma: float = 1.0,
        step: float = 0.1,
        binner: BinningScheme | None = None,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.step = step
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = BinningScheme(edges=edges)
        self.binner = binner

        self.time_emb = nn.Sequential(
            nn.Linear(2, time_embed_dim),
            nn.SiLU(),
        )
        self.core = make_mlp(in_dim + 1 + time_embed_dim, 1, hidden_dims)

    # ----------------------------------------------------------- #
    def phi(
        self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict the average velocity from ``s`` to ``t``."""
        # Ensure y has the right shape for concatenation
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        te = torch.stack([torch.cos(math.pi * s), torch.sin(math.pi * s)], dim=-1)
        te = self.time_emb(te)
        inp = torch.cat([x, y, te], dim=-1)
        return self.core(inp).squeeze(-1)

    # ----------------------------------------------------------- #
    def mf_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean-Flow regression loss implementing Eq.(★)."""
        y = y.squeeze(-1)
        B = y.size(0)
        s = torch.rand(B, device=y.device) * (1 - self.step) + self.step
        t = s - self.step

        eps = torch.randn_like(y) * self.sigma
        y_s = torch.sqrt(1 - s) * y + torch.sqrt(s) * eps

        v_s = self.phi(x, y_s, s, s)

        def f(y_in: torch.Tensor) -> torch.Tensor:
            return self.phi(x, y_in, s, s)

        jvp_val = jvp(f, y_s, v_s, create_graph=False)[1]
        target = v_s - (s - t) * jvp_val
        target = target.detach()

        v_bar_pred = self.phi(x, y_s, s, t)
        return F.mse_loss(v_bar_pred, target)

    # ----------------------------------------------------------- #
    @torch.no_grad()
    def predict_logits(
        self, x: torch.Tensor, n_particles: int = 256, steps: int = 2
    ) -> torch.Tensor:
        """Few-step Mean-Flow sampling and histogramming."""
        B = x.size(0)
        device = x.device
        y = torch.randn(B, n_particles, device=device) * self.sigma
        s = torch.ones(B * n_particles, device=device)

        for _ in range(steps):
            t = (s - self.step).clamp(min=0)
            y = y.view(-1)
            v_bar = self.phi(
                x.repeat_interleave(n_particles, 0),
                y,
                s,
                t,
            )
            y = (y - (s - t) * v_bar).view(B, n_particles)
            s = t

        idx = self.binner.to_index(y)
        K = self.binner.n_bins
        hist = torch.zeros(B, K, device=device)
        hist.scatter_add_(1, idx, torch.ones_like(idx, dtype=hist.dtype))
        probs = (hist + 0.5) / (n_particles + 0.5 * K)
        return probs.log()

    # ----------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for each output bin given x."""
        return self.predict_logits(x)

    def __call__(self, *args, **kw):
        raise RuntimeError("Use mf_loss() for training or predict_logits() for eval.")

    @classmethod
    def default_config(cls):
        from ..configs.model import ModelConfig

        return ModelConfig(
            name="mean_flow",
            params={
                "in_dim": 1,
                "hidden_dims": [128, 128],
                "time_embed_dim": 32,
                "sigma": 1.0,
                "step": 0.1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
            },
        )
