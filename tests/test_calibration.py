import torch
from outdist.calibration import register_calibrator, get_calibrator, BaseCalibrator
from outdist.configs.calibration import CalibratorConfig
from outdist.training.trainer import Trainer
from outdist.configs.trainer import TrainerConfig
from outdist.data.datasets import make_dataset
from outdist.data.binning import EqualWidthBinning
from outdist.models import get_model


@register_calibrator("dummy")
class DummyCalibrator(BaseCalibrator):
    def __init__(self, factor: float = 1.0, n_bins: int = 1) -> None:
        super().__init__()
        self.factor = factor
        self.n_bins = n_bins
        self.fit_called = False
        self.forward_called = False

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        self.forward_called = True
        return probs * self.factor

    def fit(self, probs: torch.Tensor, y: torch.Tensor) -> "DummyCalibrator":
        self.fit_called = True
        return self


def test_get_calibrator_instantiates_registered_calibrator():
    cfg = CalibratorConfig(name="dummy", params={"factor": 0.5, "n_bins": 1})
    calib = get_calibrator(cfg)
    assert isinstance(calib, DummyCalibrator)
    assert calib.factor == 0.5


def test_trainer_fits_and_uses_calibrator():
    train_ds, val_ds, test_ds = make_dataset("synthetic", n_samples=20)
    binning = EqualWidthBinning(0.0, 10.0, n_bins=10)
    calib_cfg = CalibratorConfig(name="dummy", params={"factor": 1.0})
    trainer = Trainer(TrainerConfig(max_epochs=1, batch_size=4), calibrator_cfg=calib_cfg)
    model = get_model("mlp", in_dim=1, n_bins=10, hidden_dims=[4])
    ckpt = trainer.fit(model, binning, train_ds, val_ds)
    assert isinstance(trainer.calibrator, DummyCalibrator)
    assert trainer.calibrator.fit_called

    results = trainer.evaluate(ckpt.model, test_ds, metrics=["nll"])
    assert "nll" in results
    assert trainer.calibrator.forward_called
