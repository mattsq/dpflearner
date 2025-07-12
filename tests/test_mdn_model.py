import torch
from outdist.models import get_model
from outdist.models.mdn import MixtureDensityNetwork


def test_mdn_forward_shape():
    model = get_model(
        "mdn",
        in_dim=3,
        start=0.0,
        end=1.0,
        n_bins=4,
        n_components=2,
        hidden_dims=[5],
    )
    x = torch.randn(2, 3)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_default_config_instantiates_mdn():
    cfg = MixtureDensityNetwork.default_config()
    model = get_model(cfg)
    assert isinstance(model, MixtureDensityNetwork)

