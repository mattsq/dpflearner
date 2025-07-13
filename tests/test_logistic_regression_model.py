import torch
from outdist.models import get_model
from outdist.models.logistic_regression import LogisticRegression


def test_logreg_forward_shape():
    model = get_model("logreg", in_dim=3, n_bins=4)
    x = torch.randn(2, 3)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_default_config_instantiates_logreg():
    cfg = LogisticRegression.default_config()
    model = get_model(cfg)
    assert isinstance(model, LogisticRegression)
