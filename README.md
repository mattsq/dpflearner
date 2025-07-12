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
model = get_model("mlp", in_dim=1, out_dim=10)
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
- `BootstrappedUniformBinning` – average uniform bins over bootstrap samples
- `BootstrappedQuantileBinning` – average quantile bins over bootstrap samples

## Calibration

Optional probability calibration is provided via the classes in
`outdist.calibration`. Calibrators follow a small registry similar to models and
are trained on held-out validation data by `Trainer` when a
`CalibratorConfig` is supplied. The built-in
`DirichletCalibrator` implements the method of Kull et&nbsp;al.&nbsp;2019 for
adjusting predicted categorical probabilities.
