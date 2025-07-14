import math

from outdist.training.logger import TrainingLogger, ConsoleLogger
from outdist.training.trainer import Trainer
from outdist.configs.trainer import TrainerConfig
from outdist.data.datasets import make_dataset
from outdist.data.binning import EqualWidthBinning
from outdist.models import get_model


class CountingLogger(TrainingLogger):
    def __init__(self) -> None:
        super().__init__()
        self.batches = 0
        self.epochs = 0
        self.avgs = []

    def on_batch_end(self, batch_idx: int, loss: float) -> None:
        self.batches += 1

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        self.epochs += 1
        self.avgs.append(avg_loss)


def test_trainer_uses_logger() -> None:
    train_ds, val_ds, _ = make_dataset("synthetic", n_samples=10)
    logger = CountingLogger()
    trainer = Trainer(TrainerConfig(max_epochs=1, batch_size=4), logger=logger)
    binning = EqualWidthBinning(0.0, 10.0, n_bins=10)
    model = get_model("logreg", in_dim=1, n_bins=10)
    trainer.fit(model, binning, train_ds, val_ds)

    expected_batches = math.ceil(len(train_ds) / trainer.cfg.batch_size)
    assert logger.batches == expected_batches
    assert logger.epochs == 1
    assert logger.avgs


def test_console_logger_subclass() -> None:
    assert issubclass(ConsoleLogger, TrainingLogger)

class DummyCollectLogger(TrainingLogger):
    def __init__(self):
        super().__init__()
        self.avgs = []

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        self.avgs.append(avg_loss)


def test_training_logger_computes_average():
    logger = DummyCollectLogger()
    logger.start_epoch(0, num_batches=3)
    logger.log_batch(0, 1.0)
    logger.log_batch(1, 2.0)
    logger.log_batch(2, 3.0)
    logger.end_epoch(0)
    assert logger.avgs == [2.0]

    logger.start_epoch(1, num_batches=1)
    assert logger.losses == []
