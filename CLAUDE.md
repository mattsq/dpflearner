# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is `outdist` - a Python library for modeling discrete distributions over continuous targets. The library follows a plugin-based architecture where models predict logits for fixed bins, with helper classes managing binning, evaluation, and experiment management.

## Development Commands

### Environment Setup
This project uses `uv` for Python environment management. Always activate the virtual environment before running commands:

```bash
uv venv                   # Create virtual environment (first time only)
source .venv/bin/activate # Activate virtual environment
```

Or run commands directly with uv:
```bash
uv run python -m pytest  # Run with uv (automatically uses correct environment)
uv run python -c "..."   # Run Python scripts with uv
```

### Installation and Setup
```bash
uv sync                   # Install dependencies using uv
# OR if using pip:
pip install -e .
```

### Testing
```bash
uv run pytest                    # Run all tests
uv run pytest -k "test_name"     # Run specific test
uv run pytest tests/test_*.py    # Run specific test file
```

### CLI Usage
```bash
uv run python -m outdist.cli.train --model mlp --dataset synthetic --epochs 5
uv run python -m outdist.cli.evaluate --model mlp --dataset synthetic --metrics nll accuracy
```

## Architecture

### Core Design Principles
1. **Discrete distributions over continuous targets** - First-class prediction type with `.probs`, `.cdf`, `.mean()`, `.sample()` methods
2. **Plugin-based models** - Each architecture registers itself via `@register_model("name")` decorator
3. **Declarative training** - All configuration through dataclasses/Pydantic models
4. **Stateless evaluation** - Pure functions that take y_true and DistributionPrediction, return metrics
5. **Explicit binning** - `BinningScheme` objects handle continuous ↔ discrete conversions

### Key Components

#### Models (`src/outdist/models/`)
- All models inherit from `BaseModel` and implement `forward(x) -> logits`
- Models register themselves: `@register_model("name")`
- Available models: `logreg`, `mlp`, `gaussian_ls`, `mdn`, `logistic_mixture`, `flow`, `ckde`, `quantile_rf`, `lincde`, `rfcde`, `imm_jump`, `evidential`, `transformer`

#### Binning (`src/outdist/data/binning.py`)
- `EqualWidthBinning` - evenly spaced bins
- `QuantileBinning` - quantile-based bins
- `bootstrap()` - averages bin edges across bootstrap resamples

#### Training (`src/outdist/training/`)
- `Trainer` - main training loop with `fit()` and `evaluate()` methods
- `EnsembleTrainer` - trains multiple models in parallel
- Supports calibration via `CalibratorConfig`
- Supports conformal prediction via `conformal_cfg`

#### Predictions (`src/outdist/predictions/discrete_dist.py`)
- `DistributionPrediction` - core prediction type with probability methods

### Configuration Structure
- `configs/model.py` - Model configurations
- `configs/trainer.py` - Training configurations  
- `configs/calibration.py` - Calibration configurations
- `configs/data.py` - Data configurations

## Common Patterns

### Adding a New Model
1. Create file in `src/outdist/models/`
2. Inherit from `BaseModel`
3. Add `@register_model("name")` decorator
4. Implement `forward()` method returning logits
5. Add corresponding test in `tests/`

### Training a Model
```python
from outdist import get_model, Trainer, make_dataset
from outdist.configs.trainer import TrainerConfig
from outdist.data.binning import EqualWidthBinning

train_ds, val_ds, test_ds = make_dataset("synthetic", n_samples=200)
model = get_model("mlp", in_dim=1, n_bins=10)
binning = EqualWidthBinning(0.0, 10.0, n_bins=10)

trainer = Trainer(TrainerConfig(max_epochs=5))
ckpt = trainer.fit(model, binning, train_ds, val_ds)
```

### Key Files to Understand
- `src/outdist/models/__init__.py` - Model registry
- `src/outdist/training/trainer.py` - Main training logic
- `src/outdist/data/binning.py` - Binning implementations
- `src/outdist/predictions/discrete_dist.py` - Core prediction type

## Testing Structure
- Tests mirror the `src/` directory structure
- Each model has corresponding `test_*_model.py` file
- Use `pytest` for running tests

### Testing Notes for Development
- **Import Issues**: Some models have external dependencies (like `nflows`) that may not be installed, causing import errors when running full test suite
- **Testing Strategy**: For testing new models in isolation, create standalone test files that import only the specific model components needed, avoiding the full model registry
- **Simple Component Tests**: Use `python test_file.py` with minimal imports for quick validation of new model implementations
- **Framework Integration**: Once basic functionality is verified, add proper pytest tests following the existing `test_*_model.py` pattern