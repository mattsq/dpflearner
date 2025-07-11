import torch
from outdist.data.transforms import Standardise, AddNoise, Compose


def test_standardise_fit_and_call():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t = Standardise().fit(data)
    out = t(data)
    assert torch.allclose(out.mean(dim=0), torch.zeros(2))
    assert torch.allclose(out.std(dim=0, unbiased=False), torch.ones(2))


def test_add_noise_changes_values():
    torch.manual_seed(0)
    data = torch.zeros(2)
    noise = AddNoise(std=0.5)
    out = noise(data)
    assert not torch.allclose(out, data)


def test_compose_applies_transforms():
    torch.manual_seed(0)
    t = Compose([AddNoise(std=0.0), AddNoise(std=0.0)])
    x = torch.randn(3)
    y = t(x.clone())
    assert torch.allclose(x, y)
