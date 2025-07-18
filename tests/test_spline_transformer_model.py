import torch
from outdist.models import get_model
from outdist.models.spline_transformer import SplineTransformer


def test_spline_transformer_forward_and_probs():
    model = get_model(
        "spline_transformer",
        in_dim=3,
        n_bins=5,
        d_model=16,
        n_heads=2,
        n_layers=1,
    )
    x = torch.randn(4, 3)
    logits = model(x)
    assert logits.shape == (4, 5)

    lp = torch.logsumexp(logits, dim=-1)
    assert torch.allclose(lp, torch.zeros_like(lp), atol=1e-4)


def test_default_config_instantiates_spline_transformer():
    cfg = SplineTransformer.default_config()
    model = get_model(cfg)
    assert isinstance(model, SplineTransformer)
    x = torch.randn(2, cfg.params["in_dim"])
    logits = model(x)
    assert logits.shape[1] == cfg.params["n_bins"]

