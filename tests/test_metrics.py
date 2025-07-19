import torch
from outdist.metrics import nll, accuracy, crps, METRICS_REGISTRY


def test_nll_matches_cross_entropy():
    logits = torch.tensor([[2.0, 1.0], [0.5, 0.5]])
    targets = torch.tensor([0, 1])
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.isclose(nll(logits, targets), expected)


def test_nll_accepts_float_targets():
    logits = torch.tensor([[2.0, 1.0], [0.5, 0.5]])
    targets = torch.tensor([0.0, 1.0])  # float dtype
    expected = torch.nn.functional.cross_entropy(logits, targets.long())
    assert torch.isclose(nll(logits, targets), expected)


def test_accuracy_computation():
    logits = torch.tensor([[2.0, 1.0], [0.5, 0.5]])
    targets = torch.tensor([0, 1])
    assert accuracy(logits, targets) == 0.5


def test_crps_perfect_prediction():
    """CRPS should be 0 for perfect predictions."""
    logits = torch.tensor([[10.0, -10.0, -10.0]])  # Strong prediction for bin 0
    targets = torch.tensor([0])
    assert torch.isclose(crps(logits, targets), torch.tensor(0.0), atol=1e-6)


def test_crps_uniform_distribution():
    """CRPS for uniform distribution should be reasonable."""
    logits = torch.tensor([[0.0, 0.0, 0.0]])  # Uniform over 3 bins
    targets = torch.tensor([1])  # Target is middle bin
    score = crps(logits, targets)
    # For uniform distribution over 3 bins with target in middle, CRPS ≈ 2/9
    assert torch.isclose(score, torch.tensor(2.0/9.0), atol=1e-6)


def test_crps_batch_computation():
    """CRPS should work with batches."""
    logits = torch.tensor([[10.0, -10.0, -10.0],
                           [-10.0, 10.0, -10.0],
                           [0.0, 0.0, 0.0]])
    targets = torch.tensor([0, 1, 2])
    score = crps(logits, targets)
    assert score.shape == torch.Size([])  # Should return scalar
    assert score >= 0.0  # CRPS is non-negative


def test_crps_float_targets():
    """CRPS should accept float targets and convert to long."""
    logits = torch.tensor([[2.0, 1.0], [0.5, 0.5]])
    targets = torch.tensor([0.0, 1.0])  # float dtype
    score = crps(logits, targets)
    assert score >= 0.0


def test_metrics_registry_contents():
    """METRICS_REGISTRY should expose the expected metrics."""
    assert METRICS_REGISTRY["nll"] is nll
    assert METRICS_REGISTRY["accuracy"] is accuracy
    assert METRICS_REGISTRY["crps"] is crps
    assert set(METRICS_REGISTRY.keys()) == {"nll", "accuracy", "crps"}
