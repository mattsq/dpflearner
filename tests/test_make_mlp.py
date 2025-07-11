import torch.nn as nn
from outdist.utils import make_mlp


def test_make_mlp_basic_dimensions():
    mlp = make_mlp(2, 3, hidden_dims=[4, 5])
    linears = [m for m in mlp if isinstance(m, nn.Linear)]
    assert len(linears) == 3
    assert linears[0].in_features == 2
    assert linears[0].out_features == 4
    assert linears[1].in_features == 4
    assert linears[1].out_features == 5
    assert linears[2].in_features == 5
    assert linears[2].out_features == 3


def test_make_mlp_norm_and_dropout():
    mlp = make_mlp(
        1,
        1,
        hidden_dims=2,
        activation=nn.ReLU,
        dropout=0.1,
        norm_layer=lambda d: nn.LayerNorm(d),
    )
    has_norm = any(isinstance(m, nn.LayerNorm) for m in mlp)
    has_dropout = any(isinstance(m, nn.Dropout) for m in mlp)
    assert has_norm and has_dropout
