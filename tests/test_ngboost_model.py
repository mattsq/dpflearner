import torch
import pytest

ngboost = pytest.importorskip("ngboost")

from outdist.models import get_model
from outdist.models.ngboost_model import NGBoostModel
from outdist.configs.model import ModelConfig
from outdist.data.binning import BinningScheme


def test_ngboost_smoke():
    X = torch.randn(20, 2)
    y = torch.randn(20)
    edges = torch.linspace(-1.0, 1.0, 6)
    bins = BinningScheme(edges=edges)
    model = get_model(
        "ngboost",
        start=-1.0,
        end=1.0,
        n_bins=5,
        binner=bins,
    )
    model.fit(X, y)
    logits = model.predict_logits(X)
    assert logits.shape == (20, 5)


def test_default_config_instantiates_ngboost():
    cfg = NGBoostModel.default_config()
    model = get_model(cfg)
    assert isinstance(model, NGBoostModel)
