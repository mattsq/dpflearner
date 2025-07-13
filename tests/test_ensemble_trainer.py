import itertools
import importlib
import pytest
import torch

from outdist.training.ensemble_trainer import EnsembleTrainer
from outdist.configs.trainer import TrainerConfig
from outdist.configs.model import ModelConfig
from outdist.data.datasets import make_dataset
from outdist.data.binning import EqualWidthBinning
from outdist.data.binning import LearnableBinningScheme
from outdist.models import get_model
from outdist.ensembles.average import AverageEnsemble

# Same set of model configs used in trainer tests
MODEL_CONFIGS = [
    ("mlp", {"in_dim": 1, "out_dim": 10, "hidden_dims": [4]}),
    ("logreg", {"in_dim": 1, "out_dim": 10}),
    ("gaussian_ls", {"in_dim": 1, "start": 0.0, "end": 1.0, "n_bins": 10}),
    (
        "mdn",
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
        {
            "in_dim": 1,
            "start": 0.0,
            "end": 1.0,
            "n_bins": 10,
            "x_bandwidth": 0.5,
            "y_bandwidth": 0.1,
        },
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
    *(
        [
            (
                "lincde",
                {
                    "in_dim": 1,
                    "start": 0.0,
                    "end": 1.0,
                    "n_bins": 10,
                    "basis": 6,
                    "trees": 10,
                },
            )
        ]
        if importlib.util.find_spec("lincde")
        else []
    ),
    *(
        [
            (
                "rfcde",
                {
                    "in_dim": 1,
                    "start": 0.0,
                    "end": 1.0,
                    "n_bins": 10,
                    "bandwidth": 0.1,
                    "trees": 5,
                    "kde_basis": 7,
                    "min_leaf": 2,
                },
            )
        ]
        if importlib.util.find_spec("rfcde")
        else []
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
def test_model_can_be_in_ensemble(name: str, kwargs: dict) -> None:
    model = get_model(name, **kwargs)
    ens = AverageEnsemble([model])
    assert isinstance(ens, AverageEnsemble)
    assert len(ens.models) == 1


@pytest.mark.parametrize("cfg_pair", list(itertools.combinations(MODEL_CONFIGS, 2)))
def test_ensemble_trainer_runs(cfg_pair) -> None:
    (name1, kwargs1), (name2, kwargs2) = cfg_pair
    model_cfgs = [
        ModelConfig(name=name1, params=kwargs1),
        ModelConfig(name=name2, params=kwargs2),
    ]
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=20)
    binning = EqualWidthBinning(0.0, 10.0, n_bins=10)
    trainer_cfg = TrainerConfig(max_epochs=1, batch_size=4)
    ens_trainer = EnsembleTrainer(model_cfgs, trainer_cfg, bootstrap=False, n_jobs=1)
    ensemble = ens_trainer.fit(binning, train_ds, val_ds)
    assert isinstance(ensemble, AverageEnsemble)
    assert len(ensemble.models) == 2


def test_stacked_ensemble_trainer_runs() -> None:
    model_cfgs = [
        ModelConfig(
            name="mlp", params={"in_dim": 1, "out_dim": 10, "hidden_dims": [4]}
        ),
        ModelConfig(name="logreg", params={"in_dim": 1, "out_dim": 10}),
    ]
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=20)
    binning = EqualWidthBinning(0.0, 10.0, n_bins=10)
    trainer_cfg = TrainerConfig(max_epochs=1, batch_size=4)
    ens_trainer = EnsembleTrainer(
        model_cfgs, trainer_cfg, bootstrap=False, n_jobs=1, stack=True
    )
    ensemble = ens_trainer.fit(binning, train_ds, val_ds)
    from outdist.ensembles.stacked import StackedEnsemble

    assert isinstance(ensemble, StackedEnsemble)
    assert len(ensemble.models) == 2
    assert torch.isclose(ensemble.weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_ensemble_trainer_rejects_bootstrap_with_learnable_bins() -> None:
    model_cfgs = [
        ModelConfig(name="mlp", params={"in_dim": 1, "out_dim": 10, "hidden_dims": [4]})
    ]
    train_ds, val_ds, _ = make_dataset("dummy", n_samples=20)
    binning = LearnableBinningScheme(0.0, 1.0, 5)
    trainer_cfg = TrainerConfig(max_epochs=0, batch_size=4)
    ens_trainer = EnsembleTrainer(model_cfgs, trainer_cfg, bootstrap=True, n_jobs=1)
    with pytest.raises(ValueError):
        _ = ens_trainer.fit(binning, train_ds, val_ds)


def test_ensemble_trainer_aligns_incompatible_bins() -> None:
    model_cfgs = [
        ModelConfig(
            name="gaussian_ls",
            params={"in_dim": 1, "start": 0.0, "end": 1.0, "n_bins": 10},
        ),
        ModelConfig(
            name="gaussian_ls",
            params={"in_dim": 1, "start": 0.0, "end": 1.0, "n_bins": 10},
        ),
    ]

    train_ds, val_ds, _ = make_dataset("dummy", n_samples=40)

    def factory(_y: torch.Tensor) -> LearnableBinningScheme:
        return LearnableBinningScheme(0.0, 1.0, 10)

    trainer_cfg = TrainerConfig(max_epochs=1, batch_size=8)
    ens_trainer = EnsembleTrainer(model_cfgs, trainer_cfg, bootstrap=False, n_jobs=1)
    ensemble = ens_trainer.fit(factory, train_ds, val_ds)

    edges0 = ensemble.models[0].binner.edges
    edges1 = ensemble.models[1].binner.edges
    assert torch.allclose(edges0, edges1)
    inner0 = ensemble.models[0].model.binner.edges
    inner1 = ensemble.models[1].model.binner.edges
    assert not torch.allclose(inner0, inner1)
