import torch
from torch import nn
import torch.nn.functional as F


class RQSplineHead(nn.Module):
    """Predicts a monotone rational--quadratic spline CDF."""

    def __init__(self, in_dim: int, n_bins: int, n_knots: int = 8) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.n_knots = n_knots
        out_dim = 3 * (n_knots + 1)  # widths, heights, slopes
        self.proj = nn.Linear(in_dim, out_dim)

    @staticmethod
    def _rqs(x: torch.Tensor, cumx: torch.Tensor, cumy: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Evaluate monotone rational quadratic spline at ``x``."""
        widths = cumx[..., 1:] - cumx[..., :-1]
        heights = cumy[..., 1:] - cumy[..., :-1]
        delta = heights / widths

        # Determine interval for each x
        bin_idx = torch.searchsorted(cumx[..., 1:], x)
        bin_idx = bin_idx.clamp(max=widths.size(-1) - 1)

        input_cumx = cumx.gather(-1, bin_idx)
        input_cumy = cumy.gather(-1, bin_idx)
        input_widths = widths.gather(-1, bin_idx)
        input_heights = heights.gather(-1, bin_idx)
        input_delta = delta.gather(-1, bin_idx)
        input_d = d.gather(-1, bin_idx)
        input_d_plus_one = d.gather(-1, bin_idx + 1)

        theta = (x - input_cumx) / input_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_d * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_d + input_d_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        return input_cumy + numerator / denominator

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        params = F.softplus(self.proj(h)) + 1e-4
        w, h_, d = params.chunk(3, dim=-1)  # each (B, K+1)

        # Normalise widths / heights to sum to 1
        w = w / w.sum(-1, keepdim=True)
        h_ = h_ / h_.sum(-1, keepdim=True)

        # Cumulative knot coordinates in [0,1]
        cumx = F.pad(torch.cumsum(w, dim=-1), (1, 0), value=0.0)
        cumy = F.pad(torch.cumsum(h_, dim=-1), (1, 0), value=0.0)
        d = F.pad(d, (1, 1), value=1.0)

        # Evaluate spline CDF at all bin edges
        edges = torch.linspace(0, 1, self.n_bins + 1, device=h.device)
        edges = edges.unsqueeze(0).expand(B, -1)
        F_edges = self._rqs(edges, cumx, cumy, d)

        probs = F_edges[:, 1:] - F_edges[:, :-1]
        return torch.log(probs.clamp_min(1e-8))
