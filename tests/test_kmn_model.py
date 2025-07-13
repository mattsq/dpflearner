import torch
from xtylearner.models.kmn_model import KMN
from outdist.training.trainer import Trainer
from outdist.configs.trainer import TrainerConfig
from outdist.data.binning import EqualWidthBinning
from outdist.data.datasets import make_dataset


def test_kmn_methods_shapes():
    model = KMN(x_dim=2, n_bins=4, n_kernels=5, y_min=-1.0, y_max=1.0)
    x = torch.randn(3, 2)
    y = torch.randn(3)
    logits = model.bin_logits(x)
    lp = model.log_prob(x, y)
    samples = model.sample(x, n=6)
    assert logits.shape == (3, 4)
    assert lp.shape == (3,)
    assert samples.shape == (3, 6)


def test_kmn_trains_with_trainer():
    train_ds, val_ds, _ = make_dataset("synthetic", n_samples=30)
    trainer = Trainer(TrainerConfig(max_epochs=1, batch_size=4))
    binning = EqualWidthBinning(0.0, 1.0, n_bins=5)
    model = KMN(x_dim=1, n_bins=5, n_kernels=8, y_min=0.0, y_max=1.0)
    ckpt = trainer.fit(model, binning, train_ds, val_ds)
    assert ckpt.epoch == 1
