import torch
from outdist.models import get_model
from outdist.models.iqn_model import IQNModel


def test_iqn_forward_shape():
    model = get_model(
        "iqn",
        in_dim=2,
        start=0.0,
        end=1.0,
        n_bins=5,
        hidden=16,
        K_fourier=8,
        layers=2,
    )
    x = torch.randn(3, 2)
    logits = model(x)
    assert logits.shape == (3, 5)


def test_default_config_instantiates_iqn():
    cfg = IQNModel.default_config()
    model = get_model(cfg)
    assert isinstance(model, IQNModel)
