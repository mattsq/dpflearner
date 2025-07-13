import torch
from outdist.models import get_model
from outdist.models.evidential import EvidentialNet
from outdist.losses import evidential_loss
from outdist.training.trainer import Trainer
from outdist.configs.trainer import TrainerConfig
from outdist.data.datasets import make_dataset
from outdist.data.binning import EqualWidthBinning


def test_evidential_forward_shapes():
    model = get_model("evidential", in_dim=2, n_bins=3, hidden_dims=[4, 4])
    x = torch.randn(5, 2)
    out = model(x)
    assert out["alpha"].shape == (5, 3)
    assert out["probs"].shape == (5, 3)
    assert out["logits"].shape == (5, 3)


def test_evidential_loss_uniform_prior():
    alpha = torch.tensor([[1.0, 1.0]])
    loss = evidential_loss(alpha, torch.tensor([0]), lam=0.0)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_evidential_model_trains():
    train_ds, val_ds, _ = make_dataset("synthetic", n_samples=20)
    trainer = Trainer(TrainerConfig(max_epochs=1, batch_size=4), loss_fn=evidential_loss)
    binning = EqualWidthBinning(0.0, 10.0, n_bins=10)
    model = get_model("evidential", in_dim=1, n_bins=10, hidden_dims=[4, 4])
    ckpt = trainer.fit(model, binning, train_ds, val_ds)
    assert ckpt.epoch == 1
    assert isinstance(ckpt.model, EvidentialNet)
