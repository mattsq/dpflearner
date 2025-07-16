import torch
from outdist.models.monotone_cdf_model import MonotoneCDFModel
from outdist.data.binning import BinningScheme


def test_monotone_cdf_smoke():
    Xtr = torch.randn(50, 2)
    ytr = torch.randn(50)
    Xte = torch.randn(10, 2)
    edges = torch.linspace(ytr.min(), ytr.max(), 65)
    bins = BinningScheme(edges=edges)
    model = MonotoneCDFModel(in_dim=2, binner=bins)
    logits = model(Xte)
    assert logits.shape == (10, 64)
