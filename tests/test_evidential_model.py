import torch
from outdist.models import get_model
from outdist.models.evidential import EvidentialModel


def test_evidential_forward_shape():
    model = get_model(
        "evidential",
        in_dim=2,
        start=0.0,
        end=1.0,
        n_bins=3,
        hidden=[4, 4],
    )
    x = torch.randn(5, 2)
    logits = model(x)
    assert logits.shape == (5, 3)


def test_evidential_log_prob_shape():
    model = get_model(
        "evidential",
        in_dim=1,
        start=0.0,
        end=1.0,
        n_bins=5,
    )
    x = torch.randn(4, 1)
    y = torch.randn(4, 1)
    logp = model.log_prob(x, y)
    assert logp.shape == (4,)


def test_default_config_instantiates_evidential():
    cfg = EvidentialModel.default_config()
    model = get_model(cfg)
    assert isinstance(model, EvidentialModel)
