"""
Monotone CDF Network for discrete–probability forecasting.
Torch ≥2.1 required (for nn.functional.softplus with beta argument).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from . import register_model
from ..configs.model import ModelConfig
from ..data.binning import BinningScheme
from .base import BaseModel


# ---------- helper -----------------------------------------------------------
def positive_linear(in_dim: int, out_dim: int) -> nn.Module:
    """Linear layer with weights constrained to be non-negative."""
    class _PosLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_raw = nn.Parameter(torch.empty(out_dim, in_dim))
            self.bias = nn.Parameter(torch.zeros(out_dim))
            nn.init.xavier_uniform_(self.weight_raw)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = F.softplus(self.weight_raw)
            return F.linear(x, w, self.bias)

    return _PosLinear()


# ---------- main model -------------------------------------------------------
@register_model("monotone_cdf")
class MonotoneCDFModel(BaseModel):
    """
    Implements a monotone CDF network a la Chilinski & Silva (2018) but
    discretised onto fixed bins to return logits.
    """

    # ~~~ default hyper-parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def default_config() -> ModelConfig:
        return ModelConfig(
            name="monotone_cdf",
            params=dict(
                in_dim=1,
                start=0.0,
                end=1.0,
                n_bins=10,
                hidden_dims=[64, 64],
                activation="softplus",
            ),
        )

    # ~~~ life-cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        hidden_dims: List[int] | None = None,
        *,
        binner: BinningScheme | None = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = BinningScheme(edges=edges)
        self.binner = binner

        dims: List[int] = list(hidden_dims)
        layers: List[nn.Module] = []

        in_dim_total = in_dim + 1  # +1 for y
        for h in dims:
            layers.append(positive_linear(in_dim_total, h))
            layers.append(nn.Softplus())
            in_dim_total = h
        layers.append(positive_linear(in_dim_total, 1))   # scalar g(x,y)

        self.net = nn.Sequential(*layers)

    # ~~~ forward pass helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _cdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute F̂(y|x) in [0,1].  Shapes broadcast as (N, …)."""
        inp = torch.cat([x, y], dim=-1)
        g = self.net(inp)
        return torch.sigmoid(g)

    # public API required by Trainer -----------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return per-bin logits for every sample in x.
        """
        N = x.shape[0]
        device = x.device
        edges = torch.tensor(self.binner.edges,
                             dtype=x.dtype, device=device)  # (K+1,)

        # Broadcast x to (N,K+1, x_dim) and y to (N,K+1,1)
        x_rep = x.unsqueeze(1).repeat(1, edges.numel(), 1)
        y_rep = edges.unsqueeze(0).unsqueeze(-1).expand(N, -1, 1)

        cdf = self._cdf(x_rep, y_rep)           # (N,K+1,1)
        cdf = cdf.squeeze(-1)

        probs = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(1e-12)  # (N,K)
        return probs.log()                      # logits expected downstream
