import torch
from outdist.models import get_model
from outdist.models.consistency_cde_model import ConsistencyCDEModel
from outdist.data.binning import BinningScheme


def test_consistency_smoke():
    x = torch.randn(50, 1)
    y = torch.randn(50, 1)
    binning = BinningScheme(edges=torch.linspace(-3, 3, 65))
    model = get_model("consistency_cde", in_dim=1, binner=binning)
    model.fit(x, y, epochs=1, batch_size=10)
    logits = model.predict_logits(torch.randn(8, 1))
    assert logits.shape == (8, binning.n_bins)
