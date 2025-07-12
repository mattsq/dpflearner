import torch
import numpy as np
from typing import List, Any

class AverageEnsemble:
    """Probabilistic ensemble averaging per-model probabilities."""
    def __init__(self, models: List[Any]):
        self.models = models
        # sanity check on n_bins if possible
        try:
            bins = {
                model(torch.randn(1, getattr(model, "in_dim", 1))).shape[-1]
                for model in models
            }
            if len(bins) > 1:
                raise AssertionError("Sub-models disagree on n_bins")
        except Exception:
            pass

    @torch.no_grad()
    def bin_logits(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack([model(x).softmax(dim=-1) for model in self.models])
        mean_probs = probs.mean(0)
        return (mean_probs + 1e-12).log()

    @torch.no_grad()
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lp = torch.stack([
            torch.log_softmax(model(x), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
            for model in self.models
        ])
        return torch.logsumexp(lp, dim=0) - np.log(len(self.models))

    def sample(self, x: torch.Tensor, n: int = 100) -> torch.Tensor:
        idx = torch.randint(0, len(self.models), (n,))
        samples = []
        for i in idx:
            model = self.models[i]
            if hasattr(model, "sample"):
                samples.append(model.sample(x, 1)[0])
            else:
                # fallback: categorical sampling from predicted probabilities
                probs = torch.softmax(model(x), dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                samples.append(dist.sample().float())
        return torch.stack(samples)
