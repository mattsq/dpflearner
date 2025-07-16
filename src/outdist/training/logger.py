from __future__ import annotations

from typing import List


class TrainingLogger:
    """Base class for tracking training progress."""

    def __init__(self) -> None:
        self.losses: List[float] = []
        self.num_batches: int = 0

    # ------------------------------------------------------------------
    def start_epoch(self, epoch: int, num_batches: int) -> None:
        """Reset state for a new epoch."""
        self.losses = []
        self.num_batches = num_batches

    # ------------------------------------------------------------------
    def log_batch(
        self, batch_idx: int, loss: float, metrics: dict[str, float] | None = None
    ) -> None:
        """Record statistics for a finished batch."""
        self.losses.append(loss)
        self.on_batch_end(batch_idx, loss, metrics or {})

    # ------------------------------------------------------------------
    def end_epoch(self, epoch: int) -> None:
        """Compute epoch statistics and trigger :meth:`on_epoch_end`."""
        avg = sum(self.losses) / len(self.losses) if self.losses else 0.0
        self.on_epoch_end(epoch, avg)

    # ------------------------------------------------------------------
    def on_batch_end(
        self, batch_idx: int, loss: float, metrics: dict[str, float]
    ) -> None:  # pragma: no cover - hook
        """Hook executed when a batch finishes."""
        pass

    # ------------------------------------------------------------------
    def on_epoch_end(
        self, epoch: int, avg_loss: float
    ) -> None:  # pragma: no cover - hook
        """Hook executed at the end of an epoch."""
        pass


class ConsoleLogger(TrainingLogger):
    """Simple logger that prints batch and epoch statistics to ``stdout``."""

    def on_batch_end(
        self, batch_idx: int, loss: float, metrics: dict[str, float]
    ) -> None:
        is_last = batch_idx + 1 == self.num_batches
        end = "\n" if is_last else "\r"
        parts = [f"Batch {batch_idx + 1}/{self.num_batches}"]
        parts.append(f"loss: {loss:.4f}")
        for name, value in metrics.items():
            parts.append(f"{name}: {value:.4f}")
        print(" ".join(parts), end=end, flush=True)

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
