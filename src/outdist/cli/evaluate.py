"""Command line interface for evaluating a model."""

from __future__ import annotations

import argparse

from ..configs.model import ModelConfig
from ..configs.trainer import TrainerConfig
from ..data.datasets import make_dataset
from ..models import get_model
from ..training.trainer import Trainer


def main(argv: list[str] | None = None) -> None:
    """Evaluate a model using the :class:`Trainer` utilities."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument(
        "--dataset",
        default="dummy",
        help="Dataset name (default: dummy)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["nll"],
        help="Metrics to compute (default: nll)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    args = parser.parse_args(argv)

    _, _, test_ds = make_dataset(args.dataset)
    trainer = Trainer(TrainerConfig(batch_size=args.batch_size))
    model = get_model(ModelConfig(name=args.model))
    results = trainer.evaluate(model, test_ds, metrics=args.metrics)
    for name, value in results.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()

