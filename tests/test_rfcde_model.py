import torch
import pytest

rfcde = pytest.importorskip("rfcde")
from outdist.models import get_model
from outdist.models.rfcde import RFCDEModel


def test_rfcde_forward_shape():
    model = get_model(
        "rfcde",
        in_dim=1,
        start=0.0,
        end=1.0,
        n_bins=5,
        bandwidth=0.1,
        trees=5,
        kde_basis=7,
        min_leaf=2,
    )
    x_train = torch.randn(20, 1)
    y_train = torch.rand(20)
    model.fit(x_train, y_train)
    x = torch.randn(4, 1)
    logits = model(x)
    assert logits.shape == (4, 5)


def test_default_config_instantiates_rfcde():
    cfg = RFCDEModel.default_config()
    model = get_model(cfg)
    assert isinstance(model, RFCDEModel)
