# dpflearner

`outdist` is a small library for modelling discrete distributions over continuous targets. Each model predicts logits for a fixed set of bins while helper classes manage binning, evaluation and experiment management. The design closely follows the outline in [prompt.md](prompt.md).

## Installation

Install the package in editable mode and run the tests to ensure everything works:

```bash
pip install -e .
pytest
```

## Quick start

The snippet below shows a minimal training loop:

```python
from outdist import get_model, Trainer, make_dataset, ConsoleLogger
from outdist.configs.trainer import TrainerConfig
from outdist.data.binning import EqualWidthBinning

# use the built-in dummy classification dataset or a synthetic regression one
train_ds, val_ds, test_ds = make_dataset("synthetic", n_samples=200, n_features=3)
model = get_model("mlp", in_dim=1, n_bins=10)
binning = EqualWidthBinning(0.0, 10.0, n_bins=10)

trainer = Trainer(
    TrainerConfig(max_epochs=5, batch_size=32),
    logger=ConsoleLogger(),
)
ckpt = trainer.fit(model, binning, train_ds, val_ds)
results = trainer.evaluate(ckpt.model, test_ds, metrics=["nll", "accuracy"])
print(results)
```

### Command line

Instead of writing Python code you can use the built-in CLI helpers:

```bash
python -m outdist.cli.train --model mlp --dataset synthetic --epochs 5
python -m outdist.cli.evaluate --model mlp --dataset synthetic --metrics nll accuracy
```

## Models

The following identifiers are recognised by `get_model` and the CLI:

- `logreg` – logistic regression baseline
- `mlp` – multilayer perceptron
- `gaussian_ls` – Gaussian location–scale model
- `mdn` – mixture density network
- `logistic_mixture` – mixture of logistics
- `flow` – conditional normalising flow
- `ckde` – conditional kernel density estimator
- `quantile_rf` – quantile regression forest
- `lincde` – tree-based estimator via Lindsey's method
- `rfcde` – random forest conditional density estimator
- `imm_jump` – generative model based on inductive moment matching
- `evidential` – placeholder for an evidential neural network

New architectures can register themselves via `@register_model("name")` and become immediately available.

## Binning strategies

Utilities in `outdist.data.binning` convert continuous targets to bin indices. Each scheme implements the `BinningScheme` interface with monotonically increasing edges and helpers like `to_index` and `centers`.
Implemented strategies include:

- `EqualWidthBinning` – evenly spaced edges between a start and end value
- `QuantileBinning` – edges based on quantiles of observed data
- `bootstrap` – averages bin edges across bootstrap resamples
  ```python
  binning = bootstrap(lambda s: QuantileBinning(s, 10).edges, y_train, n_bootstrap=20)
  ```
  The trainer can perform this automatically:
  ```python
  trainer = Trainer(
      TrainerConfig(max_epochs=5),
      logger=ConsoleLogger(),
      bootstrap_bins=True,
      n_bin_bootstraps=20,
  )
  ckpt = trainer.fit(model, lambda y: QuantileBinning(y, 10), train_ds, val_ds)
  ```

## Calibration

Optional probability calibration is implemented in `outdist.calibration`. Calibrators follow a small registry similar to models and are trained on held-out validation data when a `CalibratorConfig` is provided. The included `DirichletCalibrator` implements the method of Kull et al. 2019.

```python
from outdist import get_model, Trainer, make_dataset, ConsoleLogger
from outdist.configs.trainer import TrainerConfig
from outdist.configs.calibration import CalibratorConfig
from outdist.data.binning import EqualWidthBinning

train_ds, val_ds, test_ds = make_dataset("synthetic", n_samples=200)
model = get_model("mlp", in_dim=1, n_bins=10)
binning = EqualWidthBinning(0.0, 10.0, n_bins=10)

trainer = Trainer(
    TrainerConfig(max_epochs=5, batch_size=32),
    calibrator_cfg=CalibratorConfig(name="dirichlet"),
    logger=ConsoleLogger(),
)
ckpt = trainer.fit(model, binning, train_ds, val_ds)
results = trainer.evaluate(ckpt.model, test_ds, metrics=["nll"])
print(results)
```

## Conformal intervals

Passing `conformal_cfg` to `Trainer.fit` wraps the fitted model in a conformal set predictor. The resulting `CHCDSConformal` adapter exposes `contains` and `coverage` utilities.

```python
trainer = Trainer(
    TrainerConfig(max_epochs=5, batch_size=32),
    calibrator_cfg=CalibratorConfig(name="dirichlet"),
    logger=ConsoleLogger(),
)
ckpt = trainer.fit(model, binning, train_ds, val_ds, conformal_cfg={"alpha": 0.1})
print(ckpt.model.coverage(val_x, val_y))
```

## Ensembles

`EnsembleTrainer` trains several models in parallel and combines them through averaging or stacking. Bootstrapping of the training data is enabled by default for bagging-style ensembles.

```python
from outdist.training.ensemble_trainer import EnsembleTrainer
from outdist.configs.model import ModelConfig

model_cfgs = [
    ModelConfig(name="mlp", params={"in_dim": 1, "n_bins": 10, "hidden_dims": [4]}),
    ModelConfig(name="logreg", params={"in_dim": 1, "n_bins": 10}),
]
ens_trainer = EnsembleTrainer(
    model_cfgs,
    TrainerConfig(max_epochs=5),
    logger=ConsoleLogger(),
)
ensemble = ens_trainer.fit(binning, train_ds, val_ds)
```
