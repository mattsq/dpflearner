import torch
import pytest

from outdist.training.trainer import Trainer
from outdist.configs.trainer import TrainerConfig
from outdist.data.datasets import make_dataset
from outdist.data.binning import EqualWidthBinning
from outdist.models import get_model


MODEL_CONFIGS = [
    ("mlp", {"in_dim": 1, "out_dim": 10, "hidden_dims": [4]}),
    ("logreg", {"in_dim": 1, "out_dim": 10}),
    ("gaussian_ls", {"in_dim": 1, "start": 0.0, "end": 1.0, "n_bins": 10}),
    (
        "mdn",
        {"in_dim": 1, "start": 0.0, "end": 1.0, "n_bins": 10, "n_components": 2, "hidden_dims": [4]},
    ),
    (
        "logistic_mixture",
        {
            "in_dim": 1,
            "start": 0.0,
            "end": 1.0,
            "n_bins": 10,
            "n_components": 2,
            "hidden_dims": [4],
        },
    ),
    (
        "ckde",
        {"in_dim": 1, "start": 0.0, "end": 1.0, "n_bins": 10, "x_bandwidth": 0.5, "y_bandwidth": 0.1},
    ),
    (
        "quantile_rf",
        {
            "in_dim": 1,
            "start": 0.0,
            "end": 1.0,
            "n_bins": 10,
            "n_estimators": 5,
            "random_state": 0,
        },
    ),
    ("evidential", {"in_dim": 1, "n_bins": 10, "hidden_dims": [4, 4]}),
    (
        "flow",
        {
            "in_dim": 1,
            "start": 0.0,
            "end": 1.0,
            "n_bins": 10,
            "blocks": 2,
            "hidden": 8,
            "spline_bins": 4,
        },
    ),
]


@pytest.mark.parametrize("name, kwargs", MODEL_CONFIGS)
def test_model_can_train_with_trainer(name: str, kwargs: dict) -> None:
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=20)
    trainer = Trainer(TrainerConfig(max_epochs=1, batch_size=4))
    binning = EqualWidthBinning(0.0, 10.0, n_bins=10)
    model = get_model(name, **kwargs)

    ckpt = trainer.fit(model, binning, train_ds, val_ds)
    assert ckpt.epoch == 1
    assert isinstance(ckpt.model, type(model))
