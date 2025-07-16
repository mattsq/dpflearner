import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models.base import BaseModel
from .configs.model import ModelConfig
from .data.binning import BinningScheme


class TorchModel(BaseModel):
    """Simple training loop for torch models."""

    def __init__(self, cfg: ModelConfig, binning: BinningScheme) -> None:
        super().__init__()
        self.cfg = cfg
        self.binner = binning
        self.K = binning.n_bins
        self.x_dim = cfg.params.get("in_dim", 1)

    # ------------------------------------------------------------------
    def fit(self, x: torch.Tensor, y: torch.Tensor, *, epochs: int = 1, batch_size: int = 256, lr: float = 1e-3) -> "TorchModel":
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for batch in loader:
                optim.zero_grad()
                loss = self._compute_loss(batch)
                loss.backward()
                optim.step()
                self._after_batch()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

    # ------------------------------------------------------------------
    def _compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def _after_batch(self) -> None:
        pass
