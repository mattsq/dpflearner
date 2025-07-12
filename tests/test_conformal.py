import torch
from torch.utils.data import TensorDataset

from outdist.conformal.chcds import CHCDSConformal
from outdist.training.trainer import Trainer
from outdist.configs.trainer import TrainerConfig
from outdist.data.binning import EqualWidthBinning


class DummyBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        class Bins:
            edges = torch.tensor([0.0, 1.0, 2.0])
        self.bins = Bins()

    @torch.no_grad()
    def bin_logits(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), 2)

    @torch.no_grad()
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.full_like(y, torch.log(torch.tensor(0.5)))


def test_chcds_conformal_basic():
    base = DummyBaseModel()
    adapter = CHCDSConformal(base)
    x = torch.randn(20, 1)
    y = torch.rand(20)
    adapter.calibrate(x, y, alpha=0.1)
    assert adapter.delta is not None
    mask = adapter.contains(x, y)
    assert mask.dtype == torch.bool
    cov = adapter.coverage(x, y)
    assert isinstance(cov, float)


def test_trainer_applies_conformal():
    train_x = torch.randn(20, 1)
    train_y = torch.rand(20)
    val_x = torch.randn(10, 1)
    val_y = torch.rand(10)
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    binning = EqualWidthBinning(0.0, 2.0, n_bins=2)
    model = DummyBaseModel()
    trainer = Trainer(TrainerConfig(max_epochs=0))
    ckpt = trainer.fit(model, binning, train_ds, val_ds, conformal_cfg={"alpha": 0.1})
    assert isinstance(ckpt.model, CHCDSConformal)
    assert ckpt.model.delta is not None
