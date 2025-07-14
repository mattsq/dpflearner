import torch
from outdist.dirichlet import total_uncertainty, epistemic_entropy


def test_dirichlet_measures():
    alpha = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
    tu = total_uncertainty(alpha)
    expected_tu = 2 / (alpha.sum(-1) + 1)
    assert torch.allclose(tu, expected_tu)

    ee = epistemic_entropy(alpha)
    p = alpha / alpha.sum(-1, keepdim=True)
    expected_ee = (-p * p.log()).sum(-1)
    assert torch.allclose(ee, expected_ee)
