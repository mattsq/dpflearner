import torch
from outdist.models import get_model
from outdist.models.kmn_model import KMNModel, make_centres
from outdist.data.binning import BinningScheme


def test_kmn_forward_shape():
    y_train = torch.linspace(0, 1, 100)
    centres = make_centres(y_train, k=4)
    edges = torch.linspace(0.0, 1.0, 6)
    binner = BinningScheme(edges=edges)
    model = get_model(
        "kmn",
        dim_x=2,
        binner=binner,
        centres=centres,
        hidden=[16],
    )
    x = torch.randn(3, 2)
    logits = model(x)
    assert logits.shape == (3, 5)


def test_default_config_instantiates_kmn():
    cfg = KMNModel.default_config()
    y_train = torch.linspace(0, 1, 50)
    centres = make_centres(y_train, k=cfg.params.get("n_kernels", 1))
    edges = torch.linspace(0.0, 1.0, 5)
    binner = BinningScheme(edges=edges)
    model = get_model(cfg, binner=binner, centres=centres)
    assert isinstance(model, KMNModel)
