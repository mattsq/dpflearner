import torch
from outdist.models import get_model
from outdist.models.ckde import ConditionalKernelDensityEstimator


def test_ckde_forward_shape():
    model = get_model(
        "ckde",
        in_dim=1,
        start=0.0,
        end=1.0,
        n_bins=5,
        x_bandwidth=0.5,
        y_bandwidth=0.1,
    )
    x_train = torch.randn(20, 1)
    y_train = torch.rand(20)
    model.fit(x_train, y_train)
    x = torch.randn(4, 1)
    logits = model(x)
    assert logits.shape == (4, 5)


def test_default_config_instantiates_ckde():
    cfg = ConditionalKernelDensityEstimator.default_config()
    model = get_model(cfg)
    assert isinstance(model, ConditionalKernelDensityEstimator)

