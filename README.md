# dpflearner

`outdist` is a small framework for learning discrete probability
 distributions over a continuous outcome. Models predict logits for a
 fixed set of bins and training utilities handle binning, metrics and
 evaluation. The project follows the design sketched in
 [`prompt.md`](prompt.md) and exposes simple APIs for experimentation.

## Quick start

Install the package in editable mode and run the unit tests:

```bash
pip install -e .
pytest
```

A minimal training loop looks like this:

```python
from outdist import get_model, Trainer, make_dataset
from outdist.configs.trainer import TrainerConfig
from outdist.data.binning import EqualWidthBinning

train_ds, val_ds, test_ds = make_dataset("dummy", n_samples=200)
model = get_model("mlp", in_dim=1, n_bins=10)
binning = EqualWidthBinning(0.0, 10.0, n_bins=10)

trainer = Trainer(TrainerConfig(max_epochs=5, batch_size=32))
ckpt = trainer.fit(model, binning, train_ds, val_ds)
results = trainer.evaluate(ckpt.model, test_ds, metrics=["nll", "accuracy"])
print(results)
```

You can also use the command line helpers:

```bash
python -m outdist.cli.train --model mlp --dataset dummy --epochs 5
python -m outdist.cli.evaluate --model mlp --dataset dummy --metrics nll accuracy
```

## Implemented models

The following model names can be passed to `get_model` or the CLI:

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
- `evidential` – placeholder for an evidential neural network

Additional architectures can register themselves via
`@register_model("name")` and become instantly available.

## Binning schemes

Utilities in `outdist.data.binning` convert continuous targets to discrete bin
indices. All schemes inherit from `BinningScheme` which stores monotonically
increasing bin edges and exposes methods like `to_index` and `centers`.
Implemented strategies include:

- `EqualWidthBinning` – evenly spaced edges between a start and end value
- `QuantileBinning` – edges based on quantiles of observed data
- `bootstrap` – averages bin edges across bootstrap resamples
  ```python
  binning = bootstrap(lambda s: QuantileBinning(s, 10).edges, y_train, n_bootstrap=20)
  ```
  The trainer can perform this automatically:
  ```python
  trainer = Trainer(TrainerConfig(max_epochs=5), bootstrap_bins=True, n_bin_bootstraps=20)
  ckpt = trainer.fit(model, lambda y: QuantileBinning(y, 10), train_ds, val_ds)
  ```

## Calibration

Optional probability calibration is provided via the classes in
`outdist.calibration`. Calibrators follow a small registry similar to models and
are trained on held-out validation data by `Trainer` when a
`CalibratorConfig` is supplied. The built-in
`DirichletCalibrator` implements the method of Kull et&nbsp;al.&nbsp;2019 for
adjusting predicted categorical probabilities.

## Conformal intervals

The trainer can optionally wrap the fitted model in a conformal set predictor.
Passing ``conformal_cfg`` to :meth:`Trainer.fit` returns a ``CHCDSConformal``
adapter that offers ``contains`` and ``coverage`` utilities.

```python
trainer = Trainer(
    TrainerConfig(max_epochs=5, batch_size=32),
    calibrator_cfg=CalibratorConfig(name="dirichlet"),
)
ckpt = trainer.fit(model, binning, train_ds, val_ds, conformal_cfg={"alpha": 0.1})
print(ckpt.model.coverage(val_x, val_y))
```

## Ensembles

`EnsembleTrainer` trains multiple models in parallel and combines them via
averaging or stacking. Bootstrapping of the training data is enabled by default
for bagging-style ensembles.

```python
from outdist.training.ensemble_trainer import EnsembleTrainer
from outdist.configs.model import ModelConfig

model_cfgs = [
    ModelConfig(name="mlp", params={"in_dim": 1, "n_bins": 10, "hidden_dims": [4]}),
    ModelConfig(name="logreg", params={"in_dim": 1, "n_bins": 10}),
]
ens_trainer = EnsembleTrainer(model_cfgs, TrainerConfig(max_epochs=5))
ensemble = ens_trainer.fit(binning, train_ds, val_ds)
```
