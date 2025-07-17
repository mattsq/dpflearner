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
from ..losses import cross_entropy
from ..training.logger import ConsoleLogger


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and run a simple training loop."""

    parser = argparse.ArgumentParser(
        description="Train a model on synthetic or dummy data with customizable train/validation splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default 80/10/10 split
  python -m outdist.cli.train --model mlp --dataset synthetic --epochs 5

  # Custom data split with more validation data
  python -m outdist.cli.train --model mlp --dataset synthetic --epochs 5 \\
    --train-split 0.7 --val-split 0.2 --test-split 0.1

  # Train without validation (use all data for training)
  python -m outdist.cli.train --model mlp --dataset synthetic --epochs 5 --no-validation

  # Train with more samples and larger batches
  python -m outdist.cli.train --model mlp --dataset synthetic --epochs 10 \\
    --n-samples 1000 --batch-size 64
        """
    )
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
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for training (default: cpu)",
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
    parser.add_argument(
        "--bootstrap-bins",
        action="store_true",
        help="Average bin edges over bootstrap resamples",
    )
    parser.add_argument(
        "--n-bin-bootstraps",
        type=int,
        default=10,
        help="Number of bootstrap draws for binning (default: 10)",
    )
    parser.add_argument(
        "--conformal-alpha",
        type=float,
        help="Alpha level for conformal intervals",
    )
    parser.add_argument(
        "--conformal-split",
        type=float,
        help="Fraction of training data for calibration split",
    )
    parser.add_argument(
        "--conformal-tau",
        type=float,
        help="CHCDSConformal tau parameter",
    )
    parser.add_argument(
        "--conformal-mode",
        choices=["sub", "div"],
        help="CHCDSConformal scoring mode",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Total number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation during training (train on all data)",
    )

    args = parser.parse_args(argv)

    if args.no_validation:
        # Use all data for training when validation is disabled
        train_ds, _, _ = make_dataset(
            args.dataset, 
            n_samples=args.n_samples,
            splits=(1.0, 0.0, 0.0)
        )
        val_ds = None
    else:
        # Validate splits sum to 1.0
        total_split = args.train_split + args.val_split + args.test_split
        if not abs(total_split - 1.0) < 1e-6:
            raise ValueError(f"Train/val/test splits must sum to 1.0, got {total_split}")

        train_ds, val_ds, _ = make_dataset(
            args.dataset, 
            n_samples=args.n_samples,
            splits=(args.train_split, args.val_split, args.test_split)
        )
    calib_cfg = (
        CalibratorConfig(name=args.calibrate, params={"l2": args.l2})
        if args.calibrate
        else None
    )
    cfg = TrainerConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    model = get_model(ModelConfig(name=args.model))
    loss_fn = cross_entropy
    if any(
        hasattr(model, attr) for attr in ("imm_loss", "dsm_loss", "mf_loss", "quantile_loss")
    ):
        loss_fn = None
    trainer = Trainer(
        cfg,
        calibrator_cfg=calib_cfg,
        loss_fn=loss_fn,
        logger=ConsoleLogger(),
        bootstrap_bins=args.bootstrap_bins,
        n_bin_bootstraps=args.n_bin_bootstraps,
    )
    conformal_cfg = None
    if any(
        v is not None
        for v in (
            args.conformal_alpha,
            args.conformal_split,
            args.conformal_tau,
            args.conformal_mode,
        )
    ):
        conformal_cfg = {}
        if args.conformal_alpha is not None:
            conformal_cfg["alpha"] = args.conformal_alpha
        if args.conformal_split is not None:
            conformal_cfg["split"] = args.conformal_split
        if args.conformal_tau is not None:
            conformal_cfg["tau"] = args.conformal_tau
        if args.conformal_mode is not None:
            conformal_cfg["mode"] = args.conformal_mode
    binning = EqualWidthBinning(0.0, 10.0, n_bins=10)
    trainer.fit(model, binning, train_ds, val_ds, conformal_cfg=conformal_cfg)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()

