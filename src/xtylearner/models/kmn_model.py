import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal


class KMN(nn.Module):
    """Kernel Mixture Network with fixed Gaussian kernels."""

    def __init__(
        self,
        x_dim: int,
        n_bins: int,
        *,
        n_kernels: int = 64,
        y_min: float = -5.0,
        y_max: float = 5.0,
        hidden: int = 128,
        layers: int = 2,
        sigma_frac: float = 0.3,
    ) -> None:
        super().__init__()

        centres = torch.linspace(y_min, y_max, n_kernels)
        width = (centres[1] - centres[0]) * sigma_frac
        self.register_buffer("mu", centres)  # (K,)
        self.register_buffer("sigma", torch.full_like(centres, width))
        self.K = n_kernels

        modules = [nn.Linear(x_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        modules.append(nn.Linear(hidden, n_kernels))
        self.net = nn.Sequential(*modules)

        from outdist.data.binning import BinningScheme

        edges = torch.linspace(y_min, y_max, n_bins + 1)
        self.binner = BinningScheme(edges=edges)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-bin logits for ``x``."""
        return self.bin_logits(x)

    # ------------------------------------------------------------------
    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

    # ------------------------------------------------------------------
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.net(x), dim=-1)  # (B,K)
        y = y.unsqueeze(-1)
        comp = Normal(self.mu, self.sigma)
        log_pdf = comp.log_prob(y) + w.log()
        return torch.logsumexp(log_pdf, dim=-1)

    # ------------------------------------------------------------------
    def nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, y).mean()

    # ------------------------------------------------------------------
    def bin_logits(self, x: torch.Tensor) -> torch.Tensor:
        w = self._weights(x)  # (B,K)
        edges = self.binner.edges.to(x)
        comp = Normal(self.mu.view(-1, 1), self.sigma.view(-1, 1))
        diff = comp.cdf(edges[1:].view(1, -1)) - comp.cdf(edges[:-1].view(1, -1))
        probs = w @ diff
        return (probs + 1e-12).log()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, x: torch.Tensor, n: int = 100) -> torch.Tensor:
        w = self._weights(x)  # (B,K)
        idx = torch.distributions.Categorical(w).sample((n,))  # (n,B)
        mu = self.mu[idx]
        sigma = self.sigma[idx]
        y = torch.randn_like(mu) * sigma + mu
        return y.T
