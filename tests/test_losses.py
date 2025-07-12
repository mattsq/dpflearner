import torch
from outdist.losses import cross_entropy, evidential_loss


def test_cross_entropy_matches_pytorch():
    logits = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    targets = torch.tensor([1, 0])
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.isclose(cross_entropy(logits, targets), expected)


def test_evidential_loss_simple_case():
    alpha = torch.tensor([[1.0, 1.0]])
    loss = evidential_loss(alpha, torch.tensor([0]), lam=0.0)
    assert torch.isclose(loss, torch.tensor(1.0))
