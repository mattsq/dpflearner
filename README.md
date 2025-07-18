# outdist

A Python library for modeling discrete distributions over continuous targets. `outdist` provides a plugin-based architecture where models predict logits for fixed bins, enabling probabilistic predictions with well-defined probability distributions, cumulative distribution functions, and sampling capabilities.

## Core Concepts

**Discrete distributions over continuous targets** are the central prediction type in `outdist`. Instead of point predictions, models output full probability distributions that support:

- `.probs` - probability mass for each bin
- `.cdf()` - cumulative distribution function
- `.mean()` - expected value
- `.sample()` - draw samples from the distribution

**Plugin-based models** register themselves via decorators (`@register_model("name")`) and implement a simple interface: `forward(x) -> logits`. This makes adding new architectures straightforward.

**Explicit binning** handles the conversion between continuous targets and discrete bins through `BinningScheme` objects that define bin edges and provide conversion utilities.

## Installation

Install the package in editable mode and run the tests to ensure everything works:

```bash
pip install -e .
pytest
```

## Quick Start

### Basic Training Loop

```python
from outdist import get_model, Trainer, make_dataset, ConsoleLogger
from outdist.configs.trainer import TrainerConfig
from outdist.data.binning import EqualWidthBinning

# Create synthetic dataset (use ``"synthetic-hard"`` for a tougher variant)
train_ds, val_ds, test_ds = make_dataset("synthetic", n_samples=200, n_features=3)

# Set up model and binning
model = get_model("mlp", in_dim=1, n_bins=10)
binning = EqualWidthBinning(0.0, 10.0, n_bins=10)

# Train the model
trainer = Trainer(
    TrainerConfig(max_epochs=5, batch_size=32),
    logger=ConsoleLogger(),
)
ckpt = trainer.fit(model, binning, train_ds, val_ds)

# Evaluate and get distribution predictions
results = trainer.evaluate(ckpt.model, test_ds, metrics=["nll", "accuracy"])
print(results)
```

### Making Predictions

Once trained, models return `DistributionPrediction` objects with rich probabilistic interfaces:

```python
# Get predictions for test data
test_x, test_y = next(iter(test_ds))
predictions = ckpt.model(test_x)

# Access probability distributions
probs = predictions.probs  # [batch_size, n_bins]
means = predictions.mean()  # [batch_size]
samples = predictions.sample(n_samples=100)  # [batch_size, n_samples]

# Evaluate cumulative probabilities
cdf_values = predictions.cdf(test_y)  # P(Y <= y) for each target
```

### Command Line Interface

For quick experimentation, use the built-in CLI:

```bash
python -m outdist.cli --model mlp --dataset synthetic --epochs 5
# or try the harder variant
python -m outdist.cli --model mlp --dataset synthetic-hard --epochs 5
```

## Available Models

The following model identifiers are supported by `get_model()` and the CLI:

### Basic Neural Networks
- `logreg` – logistic regression baseline
- `mlp` – multilayer perceptron
- `transformer` – transformer with self-attention and distributional output

### Parametric Distribution Models
- `gaussian_ls` – Gaussian location–scale model
- `mdn` – mixture density network
- `logistic_mixture` – mixture of logistics
- `evidential` – evidential neural network with Student-T distribution

### Flow-based and Generative Models
- `flow` – conditional normalising flow
- `diffusion` – score-based diffusion model
- `consistency_cde` – one-step conditional density estimator using consistency models
- `mean_flow` – mean flow model
- `shortcut_cde` – one-to-few-step conditional density estimator via Shortcut models

### Specialized Neural Models
- `iqn` – implicit quantile network
- `monotone_cdf` – ensures monotonic cumulative distribution functions
- `kmn` – kernel mixture network
- `spline_transformer` – transformer encoder with spline-autoregressive head

### Tree-based and Ensemble Methods
- `quantile_rf` – quantile regression forest
- `lincde` – tree-based estimator via Lindsey's method
- `rfcde` – random forest conditional density estimator
- `ngboost` – natural gradient boosting

### Non-parametric Methods
- `ckde` – conditional kernel density estimator
- `imm_jump` – generative model based on inductive moment matching

### Adding New Models

New architectures can register themselves and become immediately available:

```python
from outdist.models.base import BaseModel
from outdist.models import register_model

@register_model("my_model")
class MyModel(BaseModel):
    def forward(self, x):
        # Return logits for each bin
        return self.net(x)
```

## Binning Strategies

Binning schemes convert continuous targets to discrete bins. All schemes implement the `BinningScheme` interface with monotonically increasing edges and utilities like `to_index()` and `centers()`.

### Available Strategies

- **`EqualWidthBinning`** – evenly spaced edges between start and end values
- **`QuantileBinning`** – edges based on quantiles of observed data
- **`bootstrap()`** – averages bin edges across bootstrap resamples for robustness

### Bootstrap Binning

For more robust binning, especially with small datasets:

```python
from outdist.data.binning import bootstrap, QuantileBinning

# Manual bootstrap
binning = bootstrap(lambda s: QuantileBinning(s, 10).edges, y_train, n_bootstrap=20)

# Automatic bootstrap during training
trainer = Trainer(
    TrainerConfig(max_epochs=5),
    logger=ConsoleLogger(),
    bootstrap_bins=True,
    n_bin_bootstraps=20,
)
ckpt = trainer.fit(model, lambda y: QuantileBinning(y, 10), train_ds, val_ds)
```

## Advanced Features

### Probability Calibration

Improve prediction calibration using post-hoc calibration methods. The `DirichletCalibrator` implements the method of Kull et al. 2019:

```python
from outdist.configs.calibration import CalibratorConfig

trainer = Trainer(
    TrainerConfig(max_epochs=5, batch_size=32),
    calibrator_cfg=CalibratorConfig(name="dirichlet"),
    logger=ConsoleLogger(),
)
ckpt = trainer.fit(model, binning, train_ds, val_ds)
# Calibrated predictions are automatically applied
```

### Conformal Prediction

Generate distribution-free prediction intervals with coverage guarantees:

```python
# Train with conformal prediction
ckpt = trainer.fit(
    model, binning, train_ds, val_ds, 
    conformal_cfg={"alpha": 0.1}  # 90% coverage
)

# Check coverage on validation set
test_x, test_y = next(iter(test_ds))
coverage = ckpt.model.coverage(test_x, test_y)
print(f"Empirical coverage: {coverage:.2%}")

# Check if specific predictions contain true values
contains = ckpt.model.contains(test_x, test_y)
```

### Ensemble Training

Train multiple models in parallel and combine predictions:

```python
from outdist.training.ensemble_trainer import EnsembleTrainer
from outdist.configs.model import ModelConfig

# Define ensemble components
model_cfgs = [
    ModelConfig(name="mlp", params={"in_dim": 1, "n_bins": 10, "hidden_dims": [64, 32]}),
    ModelConfig(name="mdn", params={"in_dim": 1, "n_bins": 10, "n_components": 5}),
    ModelConfig(name="logreg", params={"in_dim": 1, "n_bins": 10}),
]

# Train ensemble (with bootstrap sampling by default)
ens_trainer = EnsembleTrainer(
    model_cfgs,
    TrainerConfig(max_epochs=10),
    logger=ConsoleLogger(),
)
ensemble = ens_trainer.fit(binning, train_ds, val_ds)

# Ensemble predictions combine individual model outputs
ensemble_preds = ensemble(test_x)
```

## Common Workflows

### Custom Dataset Integration

```python
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Create data loaders
train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=32)
val_loader = DataLoader(MyDataset(X_val, y_val), batch_size=32)

# Use with existing training pipeline
model = get_model("mlp", in_dim=X_train.shape[1], n_bins=20)
binning = QuantileBinning(y_train, n_bins=20)
ckpt = trainer.fit(model, binning, train_loader, val_loader)
```

### Hyperparameter Tuning

```python
from outdist.configs.trainer import TrainerConfig

def train_with_config(hidden_dims, lr):
    model = get_model("mlp", in_dim=1, n_bins=10, hidden_dims=hidden_dims)
    trainer = Trainer(TrainerConfig(max_epochs=20, lr=lr))
    ckpt = trainer.fit(model, binning, train_ds, val_ds)
    return trainer.evaluate(ckpt.model, val_ds, metrics=["nll"])

# Grid search example
best_nll = float('inf')
for hidden_dims in [[32], [64], [32, 16]]:
    for lr in [0.01, 0.001]:
        results = train_with_config(hidden_dims, lr)
        if results["nll"] < best_nll:
            best_nll = results["nll"]
            best_config = (hidden_dims, lr)
```

### Transformer Model Usage

The transformer model uses self-attention to capture dependencies between input features:

```python
# Basic transformer configuration
model = get_model(
    "transformer",
    in_dim=5,              # Number of input features
    n_bins=20,             # Number of output bins
    d_model=64,            # Hidden dimension
    n_heads=8,             # Number of attention heads
    n_layers=2,            # Number of transformer layers
    pooling="mean",        # Pooling strategy: "mean", "max", "sum", "last"
    dropout=0.1            # Dropout rate
)

# Advanced transformer configuration
model = get_model(
    "transformer",
    in_dim=10,
    n_bins=50,
    d_model=128,
    n_heads=8,
    n_layers=4,
    d_ff=512,              # Feed-forward dimension (default: d_model * 4)
    max_seq_len=1000,      # Maximum sequence length for positional encoding
    pooling="mean"
)
```

### Model Comparison

```python
models_to_compare = ["mlp", "mdn", "transformer", "gaussian_ls", "quantile_rf"]
results = {}

for model_name in models_to_compare:
    if model_name == "transformer":
        model = get_model(model_name, in_dim=1, n_bins=10, d_model=32, n_heads=4)
    else:
        model = get_model(model_name, in_dim=1, n_bins=10)
    ckpt = trainer.fit(model, binning, train_ds, val_ds)
    results[model_name] = trainer.evaluate(
        ckpt.model, test_ds, 
        metrics=["nll", "accuracy", "coverage"]
    )

# Compare results
for model_name, metrics in results.items():
    print(f"{model_name}: NLL={metrics['nll']:.3f}, Acc={metrics['accuracy']:.3f}")
```