import torch
from outdist.data.binning import (
    BinningScheme,
    EqualWidthBinning,
    QuantileBinning,
    BootstrappedUniformBinning,
    BootstrappedQuantileBinning,
    KMeansBinning,
    BootstrappedKMeansBinning,
    bootstrap,
)


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


def test_bootstrap_uniform_binning_edges():
    torch.manual_seed(0)
    data = torch.arange(10, dtype=torch.float)
    scheme = BootstrappedUniformBinning(data, n_bins=2, n_bootstrap=2)
    expected = torch.tensor([0.5, 4.75, 9.0])
    assert torch.allclose(scheme.edges, expected)


def test_bootstrap_quantile_binning_edges():
    torch.manual_seed(0)
    data = torch.arange(1, 6, dtype=torch.float)
    scheme = BootstrappedQuantileBinning(data, n_bins=2, n_bootstrap=2)
    expected = torch.tensor([2.0, 4.0, 5.0])
    assert torch.allclose(scheme.edges, expected)


def test_kmeans_binning_edges_span():
    torch.manual_seed(0)
    data = torch.randn(50)
    scheme = KMeansBinning(data, n_bins=3, random_state=0)
    assert torch.isclose(scheme.edges[0], data.min())
    assert torch.isclose(scheme.edges[-1], data.max())
    assert scheme.n_bins == 3


def test_bootstrap_kmeans_binning_edges_shape():
    torch.manual_seed(0)
    data = torch.randn(50)
    scheme = BootstrappedKMeansBinning(data, n_bins=3, n_bootstrap=2, random_state=0)
    assert scheme.edges.numel() == 4


def test_bootstrap_utility_matches_quantile():
    torch.manual_seed(0)
    data = torch.arange(1, 6, dtype=torch.float)

    def strategy(sample: torch.Tensor) -> torch.Tensor:
        probs = torch.linspace(0, 1, 3, device=sample.device)
        return torch.quantile(sample, probs)

    scheme = bootstrap(strategy, data, n_bootstrap=2)
    torch.manual_seed(0)
    expected = BootstrappedQuantileBinning(data, n_bins=2, n_bootstrap=2)
    assert torch.allclose(scheme.edges, expected.edges)
