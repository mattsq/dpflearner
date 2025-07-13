import torch
import pytest
from outdist.training.trainer import Trainer
from outdist.configs.trainer import TrainerConfig
from outdist.data.datasets import make_dataset
from outdist.data.binning import (
    BinningScheme,
    LearnableBinningScheme,
    QuantileBinning,
    bootstrap,
)
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
    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(1, n_bins)
        self.binner = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(name="trainer_dummy", params={"n_bins": 10})


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
    model = DummyModel(n_bins=10)
    binning = DummyBinning()
    trainer.fit(model, binning, train_ds, val_ds)
    assert binning.fit_called


def test_trainer_accepts_learnable_bins():
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=10)
    trainer = Trainer(TrainerConfig(max_epochs=0))
    binner = LearnableBinningScheme(0.0, 1.0, 5)
    model = DummyModel(n_bins=5)
    trainer.fit(model, binner, train_ds, val_ds)
    assert isinstance(model.binner, LearnableBinningScheme)


def test_trainer_rejects_learnable_bins_for_nonmodule():
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=10)
    trainer = Trainer(TrainerConfig(max_epochs=0))
    binner = LearnableBinningScheme(0.0, 1.0, 5)
    model = PlainModel()
    with pytest.raises(TypeError):
        trainer.fit(model, binner, train_ds, val_ds)


def test_trainer_accepts_callable_binning():
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=10)
    trainer = Trainer(TrainerConfig(max_epochs=0))

    def factory(y: torch.Tensor) -> BinningScheme:
        start = float(y.min())
        end = float(y.max())
        return BinningScheme(edges=torch.linspace(start, end, 3))

    model = DummyModel(n_bins=2)
    trainer.fit(model, factory, train_ds, val_ds)
    assert isinstance(model.binner, BinningScheme)


def test_trainer_bootstraps_binning():
    torch.manual_seed(0)
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=20)

    def factory(y: torch.Tensor) -> BinningScheme:
        return QuantileBinning(y.to(torch.float), 3)

    y_all = torch.stack([y for _, y in train_ds])
    baseline = factory(y_all).edges

    trainer = Trainer(
        TrainerConfig(max_epochs=0), bootstrap_bins=True, n_bin_bootstraps=2
    )
    torch.manual_seed(0)
    model = DummyModel(n_bins=3)
    trainer.fit(model, factory, train_ds, val_ds)
    assert model.binner.n_bins == 3
    assert not torch.allclose(model.binner.edges, baseline)


def test_trainer_bootstrap_rejects_learnable():
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=20)

    def factory(_y: torch.Tensor) -> LearnableBinningScheme:
        return LearnableBinningScheme(0.0, 1.0, 5)

    trainer = Trainer(TrainerConfig(max_epochs=0), bootstrap_bins=True)
    model = DummyModel(n_bins=5)
    with pytest.raises(ValueError):
        trainer.fit(model, factory, train_ds, val_ds)

