import torch
from outdist.losses import cross_entropy


def test_cross_entropy_matches_pytorch():
    logits = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    targets = torch.tensor([1, 0])
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.isclose(cross_entropy(logits, targets), expected)
