import torch
import pytest
from outdist.training.trainer import Trainer
from outdist.configs.trainer import TrainerConfig
from outdist.data.datasets import make_dataset
from outdist.data.binning import BinningScheme, LearnableBinningScheme
from outdist.models import register_model, BaseModel
from outdist.configs.model import ModelConfig


class DummyBinning(BinningScheme):
    def __init__(self):
        super().__init__(edges=torch.tensor([0.0, 1.0]))
        self.fit_called = False

    def fit(self, data: torch.Tensor) -> "DummyBinning":
        self.fit_called = True
        return self


@register_model("trainer_dummy")
class DummyModel(BaseModel):
    def __init__(self, out_dim: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(1, out_dim)
        self.binner = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(name="trainer_dummy", params={"out_dim": 10})


class PlainModel:
    def to(self, device):
        return self

    def parameters(self):
        return []

    def __call__(self, x):
        return x


def test_trainer_fits_binning_before_training():
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=10)
    trainer = Trainer(TrainerConfig(max_epochs=0))
    model = DummyModel(out_dim=10)
    binning = DummyBinning()
    trainer.fit(model, binning, train_ds, val_ds)
    assert binning.fit_called


def test_trainer_accepts_learnable_bins():
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=10)
    trainer = Trainer(TrainerConfig(max_epochs=0))
    binner = LearnableBinningScheme(5, 0.0, 1.0)
    model = DummyModel(out_dim=5)
    trainer.fit(model, binner, train_ds, val_ds)
    assert isinstance(model.binner, LearnableBinningScheme)


def test_trainer_rejects_learnable_bins_for_nonmodule():
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=10)
    trainer = Trainer(TrainerConfig(max_epochs=0))
    binner = LearnableBinningScheme(5, 0.0, 1.0)
    model = PlainModel()
    with pytest.raises(TypeError):
        trainer.fit(model, binner, train_ds, val_ds)

