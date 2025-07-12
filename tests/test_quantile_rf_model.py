import torch
from outdist.models import get_model
from outdist.models.quantile_rf import QuantileRandomForest



def test_quantile_rf_forward_shape():
    model = get_model(
        "quantile_rf",
        in_dim=1,
        start=0.0,
        end=1.0,
        n_bins=5,
        n_estimators=10,
        random_state=0,
    )
    # Fit with small random data
    x_train = torch.randn(20, 1)
    y_train = torch.rand(20)
    model.fit(x_train, y_train)
    x = torch.randn(4, 1)
    logits = model(x)
    assert logits.shape == (4, 5)


def test_default_config_instantiates_quantile_rf():
    cfg = QuantileRandomForest.default_config()
    model = get_model(cfg)
    assert isinstance(model, QuantileRandomForest)


