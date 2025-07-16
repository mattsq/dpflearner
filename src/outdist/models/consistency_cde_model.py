"""
Consistency-Model CDE — one-step conditional density estimator.
Requires:  PyTorch >= 2.1
"""

from __future__ import annotations

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from ..configs.model import ModelConfig
from ..data.binning import BinningScheme
from ..base_torch import TorchModel


# ---------- utilities -------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sin-cos positional encoding for a (N,) tensor t in [0,1]."""
    device, half = t.device, dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device) * (-math.log(10_000.0) / (half - 1))
    )
    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------- main model ------------------------------------------------------
@register_model("consistency_cde")
class ConsistencyCDEModel(TorchModel):
    @staticmethod
    def default_config() -> ModelConfig:
        return ModelConfig(
            name="consistency_cde",
            params=dict(
                in_dim=1,
                start=0.0,
                end=1.0,
                n_bins=10,
                hidden_dim=128,
                n_layers=4,
                sigma_max=1.0,
                ema_decay=0.999,
                delta_max=0.05,
            ),
        )

    def __init__(
        self,
        *,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        hidden_dim: int = 128,
        n_layers: int = 4,
        sigma_max: float = 1.0,
        ema_decay: float = 0.999,
        delta_max: float = 0.05,
        binner: BinningScheme | None = None,
    ) -> None:
        cfg = ModelConfig(
            name="consistency_cde",
            params=dict(
                in_dim=in_dim,
                start=start,
                end=end,
                n_bins=n_bins,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                sigma_max=sigma_max,
                ema_decay=ema_decay,
                delta_max=delta_max,
            ),
        )
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = BinningScheme(edges=edges)
        super().__init__(cfg, binner)
        self.sigma_max = sigma_max
        self.delta_max = delta_max

        td = 64
        in_dim_total = self.x_dim + td
        hd = hidden_dim
        layers = [nn.Linear(in_dim_total, hd), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hd, hd), nn.SiLU()]
        layers += [nn.Linear(hd, self.K)]
        self.net = nn.Sequential(*layers)

        self.net_ema = nn.Sequential(*[copy.deepcopy(m) for m in self.net])
        for prm in self.net_ema.parameters():
            prm.requires_grad_(False)

        self.ema_decay = ema_decay
    def _forward_logits(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x:(N,D), t:(N,) -> logits:(N,K)"""
        temb = timestep_embedding(t, 64)
        h = torch.cat([x, temb], dim=-1)
        return self.net(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        t = x.new_full((N,), 1.0)
        return self._forward_logits(x, t)

    # --------------- training step ------------------------------------
    def _compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        N, device = x.size(0), x.device

        t = torch.rand(N, device=device)
        s = (t - torch.rand(N, device=device) * self.delta_max).clamp_min(0.0)

        noise = torch.randn_like(y) * self.sigma_max
        y_t = y + noise * t
        y_s = y + noise * s

        # Convert noisy targets to bin indices for consistency loss
        y_t_bins = self.binner.to_index(y_t.squeeze(-1))
        y_s_bins = self.binner.to_index(y_s.squeeze(-1))

        logits_s = self._forward_logits(x, s).detach()
        logits_t = self._forward_logits(x, t)

        # Consistency loss: model should predict similar distributions for nearby timesteps
        # and both should be consistent with the noisy target locations
        prob_s = F.softmax(logits_s, dim=-1)
        prob_t = F.softmax(logits_t, dim=-1)
        
        # Consistency constraint: predictions at different timesteps should be similar
        consistency_loss = F.mse_loss(prob_t, prob_s)
        
        # Target consistency: predictions should be consistent with noisy target locations
        target_loss_t = F.cross_entropy(logits_t, y_t_bins)
        target_loss_s = F.cross_entropy(logits_s, y_s_bins)
        
        # Combined loss
        loss = consistency_loss + 0.5 * (target_loss_t + target_loss_s)
        return loss

    # ------------------- EMA update -----------------------------------
    @torch.no_grad()
    def _after_batch(self) -> None:
        d = self.ema_decay
        for p_ema, p in zip(self.net_ema.parameters(), self.net.parameters()):
            p_ema.mul_(d).add_(p, alpha=1 - d)
