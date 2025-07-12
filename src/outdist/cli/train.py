"""Command line interface for launching a training run."""

from __future__ import annotations

import argparse

from ..configs.model import ModelConfig
from ..configs.trainer import TrainerConfig
from ..configs.calibration import CalibratorConfig
from ..data.datasets import make_dataset
from ..data.binning import EqualWidthBinning
from ..models import get_model
from ..training.trainer import Trainer


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and run a simple training loop."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name to train")
    parser.add_argument(
        "--dataset",
        default="dummy",
        help="Dataset name (default: dummy)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--calibrate",
        choices=["dirichlet"],
        help="Optional calibration method",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1e-3,
        help="L2 regularisation for Dirichlet calibration",
    )

    args = parser.parse_args(argv)

    train_ds, val_ds, _ = make_dataset(args.dataset)
    calib_cfg = (
        CalibratorConfig(name=args.calibrate, params={"l2": args.l2})
        if args.calibrate
        else None
    )
    trainer = Trainer(
        TrainerConfig(batch_size=args.batch_size, max_epochs=args.epochs),
        calibrator_cfg=calib_cfg,
    )
    binning = EqualWidthBinning(0.0, 10.0, n_bins=10)
    model = get_model(ModelConfig(name=args.model))
    trainer.fit(model, binning, train_ds, val_ds)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()

