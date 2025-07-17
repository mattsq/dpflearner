"""
Shortcut Model CDE – variable-step conditional density estimator.
"""

import copy, math, torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from ..configs.model import ModelConfig
from ..data.binning import BinningScheme
from ..base_torch import TorchModel


# --- helpers ----------------------------------------------------------------
def embed(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Sin-cos positional encoding for a (N,) tensor in [0,1]."""
    device, half = t.device, dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device) * (-math.log(10_000.0) / (half - 1))
    )
    v = t[:, None] * freqs[None]
    return torch.cat([torch.sin(v), torch.cos(v)], dim=-1)   # (N,dim)


# --- model ------------------------------------------------------------------
@register_model("shortcut_cde")
class ShortcutCDEModel(TorchModel):
    """One-to-few-step CDE via Shortcut Models."""

    # ~~~ default hyper-params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def default_config() -> ModelConfig:
        return ModelConfig(
            name="shortcut_cde",
            params=dict(
                in_dim=1,
                start=-3.0,
                end=3.0,
                n_bins=64,
                hidden_dim=128,
                n_layers=4,
                sigma_max=1.0,
                lambda_sc=1.0,
                ema_decay=0.999,
                max_d=0.1,
            ),
        )

    # ~~~ ctor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(
        self,
        *,
        in_dim: int = 1,
        start: float = -3.0,
        end: float = 3.0,
        n_bins: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 4,
        sigma_max: float = 1.0,
        lambda_sc: float = 1.0,
        ema_decay: float = 0.999,
        max_d: float = 0.1,
        binner: BinningScheme | None = None,
    ) -> None:
        cfg = ModelConfig(
            name="shortcut_cde",
            params=dict(
                in_dim=in_dim,
                start=start,
                end=end,
                n_bins=n_bins,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                sigma_max=sigma_max,
                lambda_sc=lambda_sc,
                ema_decay=ema_decay,
                max_d=max_d,
            ),
        )
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = BinningScheme(edges=edges)
        super().__init__(cfg, binner)
        self.sigma_max, self.lambda_sc = sigma_max, lambda_sc
        td, dd = 64, 32
        in_dim_total = self.x_dim + td + dd
        hd = hidden_dim

        layers = [nn.Linear(in_dim_total, hd), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hd, hd), nn.SiLU()]
        layers += [nn.Linear(hd, self.K)]
        self.net = nn.Sequential(*layers)

        self.net_ema = copy.deepcopy(self.net)
        for q in self.net_ema.parameters():
            q.requires_grad_(False)
        self.ema_decay = ema_decay
        self.max_d = max_d

    # ~~~ core forward (student) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _forward_logits(self, x, t, d):
        """Shapes: x(N,D) t(N,) d(N,) -> logits(N,K)"""
        h = torch.cat([x, embed(t), embed(d, 32)], dim=-1)
        return self.net(h)

    # --- public inference call ----------------------------------------------
    def forward(self, x):
        """One-step predict: t=1, d=1."""
        N = x.size(0)
        t = x.new_full((N,), 1.0)
        d = x.new_full((N,), 1.0)
        return self._forward_logits(x, t, d)

    # ~~~ training loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_loss(self, batch):
        x, y = batch
        N, dev = x.size(0), x.device

        # sample timestep and step-size
        t = torch.rand(N, device=dev)
        d = torch.rand(N, device=dev) * self.max_d
        d = torch.minimum(d, t)
        t_prev = t - d

        # additive noise
        noise = torch.randn_like(y) * self.sigma_max
        y_t     = y + noise * t
        y_prev  = y + noise * t_prev

        # ---- flow-matching target (teacher at t_prev) ----------------------
        logits_prev = self._forward_logits(x, t_prev, torch.zeros_like(d)).detach()
        logits_student = self._forward_logits(x, t, d)

        prob_prev, prob_student = F.softmax(logits_prev, -1), F.softmax(logits_student, -1)
        loss_flow = F.mse_loss(prob_student, prob_prev)

        # ---- self-consistency: two small steps vs one big ------------------
        logits_half = self._forward_logits(x, t, d * 0.5).detach()
        logits_half2 = self._forward_logits(x, t - 0.5*d, d * 0.5).detach()
        target_sc = 0.5 * (F.softmax(logits_half, -1) + F.softmax(logits_half2, -1))

        logits_big = self._forward_logits(x, t, d)
        loss_sc = F.mse_loss(F.softmax(logits_big, -1), target_sc)

        return loss_flow + self.lambda_sc * loss_sc

    # ~~~ EMA update after each batch ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @torch.no_grad()
    def _after_batch(self):
        d = self.ema_decay
        for p_t, p in zip(self.net_ema.parameters(), self.net.parameters()):
            p_t.mul_(d).add_(p, alpha=1 - d)

