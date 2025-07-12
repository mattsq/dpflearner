import torch
import pytest

lincde = pytest.importorskip("lincde")
from outdist.models import get_model
from outdist.models.lincde import LinCDEModel


def test_lincde_forward_shape():
    model = get_model(
        "lincde",
        in_dim=1,
        start=0.0,
        end=1.0,
        n_bins=5,
        basis=6,
        trees=10,
    )
    x_train = torch.randn(20, 1)
    y_train = torch.rand(20)
    model.fit(x_train, y_train)
    x = torch.randn(4, 1)
    logits = model(x)
    assert logits.shape == (4, 5)


def test_default_config_instantiates_lincde():
    cfg = LinCDEModel.default_config()
    model = get_model(cfg)
    assert isinstance(model, LinCDEModel)
