import torch
from outdist.models import get_model
from outdist.models.flow_cde import FlowCDE


def test_flow_forward_shape():
    model = get_model(
        "flow",
        in_dim=2,
        start=0.0,
        end=1.0,
        n_bins=5,
        blocks=2,
        hidden=8,
        spline_bins=4,
    )
    x = torch.randn(3, 2)
    logits = model(x)
    assert logits.shape == (3, 5)


def test_default_config_instantiates_flow():
    cfg = FlowCDE.default_config()
    model = get_model(cfg)
    assert isinstance(model, FlowCDE)
