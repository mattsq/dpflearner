import torch
from outdist.models import get_model
from outdist.models.gaussian_ls import GaussianLocationScale


def test_gaussian_ls_forward_shape():
    model = get_model("gaussian_ls", in_dim=2, start=0.0, end=1.0, n_bins=5)
    x = torch.randn(4, 2)
    logits = model(x)
    assert logits.shape == (4, 5)


def test_default_config_instantiates_gaussian_ls():
    cfg = GaussianLocationScale.default_config()
    model = get_model(cfg)
    assert isinstance(model, GaussianLocationScale)
