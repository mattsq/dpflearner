from __future__ import annotations
import math
import torch
from torch import nn

from . import register_model
from ..utils import make_mlp
from ..data.binning import BinningScheme


def laplace_mmd2(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Unbiased MMD^2 with Laplace kernel k(u,v)=exp(-|u-v|/gamma)."""
    D_xx = torch.cdist(x[:, None], x[:, None], p=1)
    D_yy = torch.cdist(y[:, None], y[:, None], p=1)
    D_xy = torch.cdist(x[:, None], y[:, None], p=1)
    m, n = x.size(0), y.size(0)
    K_xx = (-D_xx / gamma).exp()
    K_yy = (-D_yy / gamma).exp()
    K_xy = (-D_xy / gamma).exp()
    return (
        (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1))
        + (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1))
        - 2 * K_xy.mean()
    )


@register_model("imm_jump")
class IMMJumpNet(nn.Module):
    """Single-step (s->t) mapping network for Inductive-Moment-Matching."""

    def __init__(
        self,
        in_dim: int = 1,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        time_embed_dim: int = 64,
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

        self.time_emb = nn.Linear(2, time_embed_dim)
        self.core = make_mlp(in_dim + 1 + time_embed_dim, 1, hidden_dims)

    def forward(
        self,
        x: torch.Tensor,
        y_noisy: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return a cleaner scalar prediction y_hat_t."""
        y_noisy = y_noisy.squeeze(-1)
        te = torch.stack([torch.cos(math.pi * s), torch.sin(math.pi * s)], dim=-1)
        te = self.time_emb(te)
        inp = torch.cat([x, y_noisy.unsqueeze(-1), te], dim=-1)
        return self.core(inp).squeeze(-1)

    def imm_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the IMM MMD loss for a batch."""
        y = y.squeeze(-1)
        B = y.size(0)
        s = torch.rand(B, device=y.device) * (1 - self.step) + self.step
        t = s - self.step

        eps = torch.randn_like(y) * self.sigma
        y_s_real = torch.sqrt(1 - s) * y + torch.sqrt(s) * eps

        use_model = torch.rand(B, device=y.device) < 0.5
        if use_model.any():
            with torch.no_grad():
                y_tmp = torch.randn_like(y) * self.sigma
                curr_s = torch.ones_like(s)
                for _ in range(int(1 / self.step)):
                    curr_t = curr_s - self.step
                    y_tmp = self.forward(x, y_tmp, curr_s, curr_t)
                    curr_s = curr_t
            y_s_real[use_model] = y_tmp[use_model]

        y_hat_t = self.forward(x, y_s_real, s, t)
        y_t_true = y + 1e-3 * torch.randn_like(y)
        return laplace_mmd2(y_hat_t, y_t_true)

    @torch.no_grad()
    def predict_logits(
        self, x: torch.Tensor, n_particles: int = 256, steps: int = 4
    ) -> torch.Tensor:
        B = x.size(0)
        device = x.device
        y = torch.randn(B, n_particles, device=device) * self.sigma
        s = torch.ones(B * n_particles, device=device)
        for _ in range(steps):
            t = s - self.step
            y = self.forward(
                x.repeat_interleave(n_particles, 0),
                y.flatten(),
                s,
                t,
            ).view(B, n_particles)
            s = t.clamp(min=0)

        idx = self.binner.to_index(y)
        K = self.binner.n_bins
        hist = torch.zeros(B, K, device=device)
        hist.scatter_add_(1, idx, torch.ones_like(idx, dtype=hist.dtype))
        probs = (hist + 0.5) / (n_particles + 0.5 * K)
        return probs.log()

    def __call__(self, x: torch.Tensor):
        raise RuntimeError(
            "Call imm_loss() during training or predict_logits() for inference."
        )

    @classmethod
    def default_config(cls):
        from ..configs.model import ModelConfig

        return ModelConfig(
            name="imm_jump",
            params={
                "in_dim": 1,
                "step": 0.1,
                "sigma": 1.0,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
            },
        )
