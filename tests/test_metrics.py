import torch
from outdist.metrics import nll, accuracy


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
