import torch
from outdist.models import get_model
from outdist.models.evidential import EvidentialModel
from outdist.losses import evidential_loss
from outdist.training.trainer import Trainer
from outdist.configs.trainer import TrainerConfig
from outdist.data.datasets import make_dataset
from outdist.data.binning import EqualWidthBinning


def test_evidential_forward_shapes():
    model = get_model(
        "evidential", in_dim=2, start=0.0, end=1.0, n_bins=3, hidden_dims=[4, 4]
    )
    x = torch.randn(5, 2)
    logits = model(x)
    assert logits.shape == (5, 3)
    y = torch.randn(5, 1)
    logp = model.log_prob(x, y)
    assert logp.shape == (5,)


def test_evidential_loss_uniform_prior():
    alpha = torch.tensor([[1.0, 1.0]])
    loss = evidential_loss(alpha, torch.tensor([0]), lam=0.0)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_evidential_model_trains():
    train_ds, val_ds, _ = make_dataset("synthetic", n_samples=20)
    trainer = Trainer(TrainerConfig(max_epochs=1, batch_size=4))
    binning = EqualWidthBinning(0.0, 1.0, n_bins=10)
    model = get_model(
        "evidential", in_dim=1, start=0.0, end=1.0, n_bins=10, hidden_dims=[4, 4]
    )
    ckpt = trainer.fit(model, binning, train_ds, val_ds)
    assert ckpt.epoch == 1
    assert isinstance(ckpt.model, EvidentialModel)
