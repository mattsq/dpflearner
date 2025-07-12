import torch
from outdist.models import get_model
from outdist.models.diffusion_cde import DiffusionCDE


def test_diffusion_forward_shape():
    model = get_model(
        "diffusion",
        in_dim=2,
        start=0.0,
        end=1.0,
        n_bins=5,
        sigma_min=1e-3,
        sigma_max=1.0,
        hidden=16,
        layers=2,
        mc_bins=32,
    )
    x = torch.randn(3, 2)
    logits = model(x)
    assert logits.shape == (3, 5)


def test_default_config_instantiates_diffusion():
    cfg = DiffusionCDE.default_config()
    model = get_model(cfg)
    assert isinstance(model, DiffusionCDE)

