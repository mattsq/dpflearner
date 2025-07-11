import torch
from outdist.data.binning import BinningScheme, EqualWidthBinning, QuantileBinning


def test_equal_width_to_index():
    scheme = EqualWidthBinning(0.0, 3.0, n_bins=3)
    y = torch.tensor([-0.5, 0.0, 0.5, 1.0, 2.9, 3.0])
    idx = scheme.to_index(y)
    assert idx.tolist() == [0, 0, 0, 1, 2, 2]


def test_quantile_binning_edges():
    data = torch.linspace(0, 1, 100)
    scheme = QuantileBinning(data, n_bins=4)
    # Edges should span the min and max of the data
    assert torch.isclose(scheme.edges[0], data.min())
    assert torch.isclose(scheme.edges[-1], data.max())
    # Number of bins should be n_bins
    assert scheme.n_bins == 4
