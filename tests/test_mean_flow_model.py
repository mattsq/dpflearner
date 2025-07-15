import torch
from outdist.models import get_model
from outdist.models.mean_flow import MeanFlow
from outdist.data.binning import EqualWidthBinning
from outdist.training.trainer import Trainer
from outdist.configs.trainer import TrainerConfig


def test_predict_logits_shape():
    binning = EqualWidthBinning(0.0, 1.0, n_bins=5)
    model = get_model("mean_flow", in_dim=2, step=0.2)
    model.binner = binning
    x = torch.randn(3, 2)
    logits = model.predict_logits(x, steps=1)
    assert logits.shape == (3, 5)


def test_trainer_runs_with_mean_flow():
    x = torch.randn(20, 2)
    y = torch.randn(20, 1)
    dataset = torch.utils.data.TensorDataset(x, y)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [16, 4])
    binning = EqualWidthBinning(0.0, 1.0, n_bins=5)
    trainer = Trainer(TrainerConfig(max_epochs=1, batch_size=4), loss_fn=None)
    model = get_model("mean_flow", in_dim=2, step=0.2)
    ckpt = trainer.fit(model, binning, train_ds, val_ds)
    assert ckpt.epoch == 1
    assert isinstance(ckpt.model, MeanFlow)
