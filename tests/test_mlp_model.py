import torch
from outdist.models import get_model
from outdist.models.mlp import MLP


def test_mlp_forward_shape():
    model = get_model("mlp", in_dim=2, n_bins=3, hidden_dims=[4])
    x = torch.randn(5, 2)
    logits = model(x)
    assert logits.shape == (5, 3)


def test_default_config_instantiates_mlp():
    cfg = MLP.default_config()
    model = get_model(cfg)
    assert isinstance(model, MLP)

