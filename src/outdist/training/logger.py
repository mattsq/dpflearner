from __future__ import annotations

from typing import Dict, List


class TrainingLogger:
    """Base class for tracking training progress."""

    def __init__(self) -> None:
        self.losses: List[float] = []
        self.metrics: Dict[str, List[float]] = {}
        self.num_batches: int = 0

    # ------------------------------------------------------------------
    def start_epoch(self, epoch: int, num_batches: int) -> None:
        """Reset state for a new epoch."""
        self.losses = []
        self.metrics = {}
        self.num_batches = num_batches

    # ------------------------------------------------------------------
    def log_batch(
        self, batch_idx: int, loss: float, metrics: dict[str, float] | None = None
    ) -> None:
        """Record statistics for a finished batch."""
        self.losses.append(loss)
        if metrics:
            for name, value in metrics.items():
                self.metrics.setdefault(name, []).append(value)
        self.on_batch_end(batch_idx, loss, metrics or {})

    # ------------------------------------------------------------------
    def end_epoch(self, epoch: int) -> None:
        """Compute epoch statistics and trigger :meth:`on_epoch_end`."""
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0.0
        avg_metrics = {
            name: sum(values) / len(values) for name, values in self.metrics.items()
        }
        self.on_epoch_end(epoch, avg_loss, avg_metrics)

    # ------------------------------------------------------------------
    def on_batch_end(
        self, batch_idx: int, loss: float, metrics: dict[str, float]
    ) -> None:  # pragma: no cover - hook
        """Hook executed when a batch finishes."""
        pass

    # ------------------------------------------------------------------
    def on_epoch_end(
        self, epoch: int, avg_loss: float, metrics: dict[str, float]
    ) -> None:  # pragma: no cover - hook
        """Hook executed at the end of an epoch."""
        pass


class ConsoleLogger(TrainingLogger):
    """Simple logger that prints batch and epoch statistics to ``stdout``."""

    def __init__(self) -> None:
        super().__init__()
        self._is_single_step = False

    def start_epoch(self, epoch: int, num_batches: int) -> None:
        """Reset state for a new epoch."""
        super().start_epoch(epoch, num_batches)
        # Detect if this is a single-step model (1 epoch, 1 batch)
        self._is_single_step = epoch == 1 and num_batches == 1

    def on_batch_end(
        self, batch_idx: int, loss: float, metrics: dict[str, float]
    ) -> None:
        # For single-step models, show a simple completion message
        if self._is_single_step:
            if "note" in metrics:
                print(metrics["note"])
            else:
                print("Model fitting complete")
            return
            
        is_last = batch_idx + 1 == self.num_batches
        end = "\n" if is_last else "\r"
        parts = [f"Batch {batch_idx + 1}/{self.num_batches}"]
        parts.append(f"loss: {loss:.4f}")
        for name, value in metrics.items():
            if name != "note":  # Skip the note field in regular batch output
                parts.append(f"{name}: {value:.4f}")
        print(" ".join(parts), end=end, flush=True)

    def on_epoch_end(self, epoch: int, avg_loss: float, metrics: dict[str, float]) -> None:
        # For single-step models, don't print epoch summary since we already printed completion
        if self._is_single_step:
            return
            
        parts = [f"Epoch {epoch} average loss: {avg_loss:.4f}"]
        for name, value in metrics.items():
            parts.append(f"{name}: {value:.4f}")
        
        # Include validation metrics if available
        if hasattr(self, '_val_metrics') and self._val_metrics:
            for name, value in self._val_metrics.items():
                parts.append(f"{name}: {value:.4f}")
        
        print(" ".join(parts))
