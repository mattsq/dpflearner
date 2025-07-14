from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..data.binning import BinningScheme
from .base import BaseModel
from ..configs.model import ModelConfig
from . import register_model

__all__ = ["KMNModel", "make_centres"]


# ------------------------------------------------------------------
# Helper to choose kernel centres
# ------------------------------------------------------------------
def make_centres(y_train: torch.Tensor, k: int, *, method: str = "quantile") -> torch.Tensor:
    if method == "quantile":
        qs = torch.linspace(0.0, 1.0, k + 2, device=y_train.device)[1:-1]
        return torch.quantile(y_train.flatten(), qs)
    raise ValueError(f"Unknown centre initialisation '{method}'")


class _KMNHead(nn.Module):
    """Neural network head predicting mixture weights and a global scale."""

    def __init__(self, in_dim: int, centres: torch.Tensor, *, hidden=(128, 128), log_sigma_init: float = -0.5) -> None:
        super().__init__()
        self.register_buffer("centres", centres)
        self.log_sigma = nn.Parameter(torch.tensor(log_sigma_init))

        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.feature = nn.Sequential(*layers)
        self.out = nn.Linear(last, centres.numel())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.out(self.feature(x))
        weights = F.softmax(logits, dim=-1)
        return weights, self.log_sigma.exp()


@register_model("kmn")
class KMNModel(BaseModel):
    """Kernel-Mixture Network producing logits over fixed bins."""

    def __init__(
        self,
        dim_x: int = 1,
        binner: BinningScheme | None = None,
        *,
        centres: torch.Tensor | None = None,
        n_kernels: int = 64,
        hidden=(128, 128),
        log_sigma_init: float = -0.5,
    ) -> None:
        super().__init__()
        if centres is None:
            centres = torch.linspace(0.0, 1.0, n_kernels)
        if binner is None:
            edges = torch.linspace(0.0, 1.0, 11)
            binner = BinningScheme(edges=edges)
        self.binner = binner
        self.head = _KMNHead(dim_x, centres, hidden=hidden, log_sigma_init=log_sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, sigma = self.head(x)
        dist = Normal(self.head.centres, sigma)
        edges = self.binner.edges.to(x)
        cdf = dist.cdf(edges.unsqueeze(-1)).transpose(0, 1)
        p_bin = (cdf[..., 1:] - cdf[..., :-1]) * weights.unsqueeze(-1)
        logits = (p_bin.sum(dim=1) + 1e-12).log()
        return logits

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="kmn",
            params={"dim_x": 1, "hidden": [128, 128], "n_kernels": 64},
        )
