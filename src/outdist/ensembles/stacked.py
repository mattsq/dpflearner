import torch
import numpy as np
from torch import nn
from typing import List, Any

class StackedEnsemble(nn.Module):
    """Linear stacking ensemble with learned weights."""

    def __init__(self, models: List[Any], weights: torch.Tensor) -> None:
        super().__init__()
        self.models = models
        self.register_buffer("weights", weights)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def bin_logits(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack([
            model(x).softmax(dim=-1) for model in self.models
        ])  # (K,N,B)
        mean = torch.einsum("k,knb->nb", self.weights, probs)
        return (mean + 1e-12).log()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lp = torch.stack([
            torch.log_softmax(model(x), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
            for model in self.models
        ])  # (K,N)
        return torch.logsumexp(lp + self.weights.log()[:, None], dim=0) - torch.log(self.weights.sum())

    # ------------------------------------------------------------------
    def sample(self, x: torch.Tensor, n: int = 100) -> torch.Tensor:
        idx = torch.multinomial(self.weights, n, replacement=True)
        samples = []
        for i in idx:
            model = self.models[i]
            if hasattr(model, "sample"):
                samples.append(model.sample(x, 1)[0])
            else:
                probs = torch.softmax(model(x), dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                samples.append(dist.sample().float())
        return torch.stack(samples)

# ----------------------------------------------------------------------

def learn_weights(models: List[Any], X_val: np.ndarray, y_val: np.ndarray, l2: float = 1e-3) -> torch.Tensor:
    """Return non-negative weights summing to one for stacking."""
    with torch.no_grad():
        X_tensor = torch.as_tensor(X_val)
        P = torch.stack([
            model(X_tensor).softmax(dim=-1) for model in models
        ])  # (K,N,B)
        y_idx = torch.as_tensor(y_val).long()
        y_one = torch.zeros_like(P[0]).scatter(1, y_idx[:, None], 1)

    w = torch.full((len(models),), 1 / len(models), requires_grad=True)
    optim = torch.optim.LBFGS([w], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        optim.zero_grad()
        weights = torch.softmax(w, 0)
        p_hat = torch.einsum("k,knb->nb", weights, P)
        loss = -(y_one * p_hat.log()).sum(1).mean() + l2 * ((weights - 1/len(models)) ** 2).mean()
        loss.backward()
        return loss

    optim.step(closure)
    return torch.softmax(w.detach(), 0)
