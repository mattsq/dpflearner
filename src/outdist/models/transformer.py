"""Transformer model with distributional output."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .base import BaseModel
from ..configs.model import ModelConfig
from ..utils import make_mlp
from . import register_model
from ..data import binning as binning_scheme


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Linear transformations and split into heads
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attn_output)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer inputs."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model by only applying cosine to available positions
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


@register_model("transformer")
class TransformerDistribution(BaseModel):
    """Transformer model with distributional output.
    
    This model treats each input feature as a sequence element and uses
    self-attention to model dependencies between features before predicting
    the distribution over target bins.
    """

    def __init__(
        self,
        in_dim: int = 1,
        start: float = 0.0,
        end: float = 1.0,
        n_bins: int = 10,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        pooling: str = "mean",
        *,
        binner: binning_scheme.BinningScheme | None = None,
    ) -> None:
        super().__init__()
        
        if d_ff is None:
            d_ff = d_model * 4
            
        self.in_dim = in_dim
        self.d_model = d_model
        self.pooling = pooling
        
        # Input embedding - project each feature to d_model dimensions
        self.input_embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection to bin logits
        self.output_projection = nn.Linear(d_model, n_bins)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Binning scheme
        if binner is None:
            edges = torch.linspace(start, end, n_bins + 1)
            binner = binning_scheme.BinningScheme(edges=edges)
        self.binner = binner
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_features = x.shape
        
        # Reshape input to treat each feature as a sequence element
        # x: (batch_size, n_features) -> (batch_size, n_features, 1)
        x = x.unsqueeze(-1)
        
        # Input embedding: (batch_size, n_features, 1) -> (batch_size, n_features, d_model)
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (n_features, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, n_features, d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Pool across sequence dimension
        if self.pooling == "mean":
            x = x.mean(dim=1)  # (batch_size, d_model)
        elif self.pooling == "max":
            x = x.max(dim=1)[0]  # (batch_size, d_model)
        elif self.pooling == "sum":
            x = x.sum(dim=1)  # (batch_size, d_model)
        elif self.pooling == "last":
            x = x[:, -1, :]  # (batch_size, d_model)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Project to bin logits
        logits = self.output_projection(x)  # (batch_size, n_bins)
        
        return logits

    @classmethod
    def default_config(cls) -> ModelConfig:
        return ModelConfig(
            name="transformer",
            params={
                "in_dim": 1,
                "start": 0.0,
                "end": 1.0,
                "n_bins": 10,
                "d_model": 64,
                "n_heads": 8,
                "n_layers": 2,
                "d_ff": None,  # Will default to d_model * 4
                "dropout": 0.1,
                "max_seq_len": 1000,
                "pooling": "mean",
            },
        )