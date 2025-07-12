import torch
from outdist.models import get_model
from outdist.models.logistic_mixture import LogisticMixture


def test_logistic_mixture_forward_shape():
    model = get_model(
        "logistic_mixture",
        in_dim=2,
        start=0.0,
        end=1.0,
        n_bins=5,
        n_components=2,
        hidden_dims=[4],
    )
    x = torch.randn(3, 2)
    logits = model(x)
    assert logits.shape == (3, 5)


def test_default_config_instantiates_logistic_mixture():
    cfg = LogisticMixture.default_config()
    model = get_model(cfg)
    assert isinstance(model, LogisticMixture)
