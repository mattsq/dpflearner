import torch
import math

from outdist.models import get_model
from outdist.ensembles.average import AverageEnsemble
from outdist.ensembles.stacked import StackedEnsemble, learn_weights


def _dummy_models(n=2, in_dim=1, n_bins=3):
    return [get_model("logreg", in_dim=in_dim, n_bins=n_bins) for _ in range(n)]


def test_average_ensemble_methods():
    models = _dummy_models()
    ens = AverageEnsemble(models)
    x = torch.randn(4, 1)
    y = torch.randint(0, 3, (4,))

    logits = ens.bin_logits(x)
    assert logits.shape == (4, 3)

    lp = ens.log_prob(x, y)
    expected = torch.logsumexp(
        torch.stack([
            torch.log_softmax(m(x), dim=-1).gather(-1, y[:, None]).squeeze(-1)
            for m in models
        ]),
        dim=0,
    ) - math.log(len(models))
    assert torch.allclose(lp, expected)

    samples = ens.sample(x[:1], n=5)
    assert samples.shape[0] == 5


def test_stacked_ensemble_methods_and_weights():
    models = _dummy_models()
    weights = torch.tensor([0.6, 0.4])
    ens = StackedEnsemble(models, weights)
    x = torch.randn(3, 1)
    y = torch.randint(0, 3, (3,))

    logits = ens.bin_logits(x)
    expected_probs = torch.stack([m(x).softmax(-1) for m in models])
    expected = torch.einsum("k,knb->nb", weights, expected_probs)
    expected_logits = (expected + 1e-12).log()
    assert torch.allclose(logits, expected_logits)

    lp = ens.log_prob(x, y)
    lps = torch.stack([
        torch.log_softmax(m(x), dim=-1).gather(-1, y[:, None]).squeeze(-1)
        for m in models
    ])
    expected_lp = torch.logsumexp(lps + weights.log()[:, None], dim=0) - math.log(weights.sum())
    assert torch.allclose(lp, expected_lp)

    samples = ens.sample(x[:1], n=4)
    assert samples.shape[0] == 4



def test_learn_weights():
    models = _dummy_models(in_dim=1)
    X_val = torch.randn(5, 1)
    y_val = torch.randint(0, 3, (5,))

    w = learn_weights(models, X_val.numpy(), y_val.numpy(), l2=1e-2)
    assert w.shape[0] == len(models)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(w >= 0)
