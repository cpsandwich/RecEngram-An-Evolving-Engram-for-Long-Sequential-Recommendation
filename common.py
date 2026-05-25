"""Shared building blocks for all models.

PositionalEncoding, MultiHeadAttention, TransformerBlock, FeedForward.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, hidden_dim: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() *
                             -(math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d)"""
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Standard multi-head scaled dot-product attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """B: batch, Lq: query length, Lk: key length, d: hidden_dim, H: heads"""
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        q = self.W_q(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Lq, hd)
        k = self.W_k(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, Lq, Lk)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn_w = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn_w, v)   # (B, H, Lq, hd)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.hidden_dim)
        return self.W_o(out)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer encoder block: self-attn + FFN, pre-norm style."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForward(hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask))
        x = x + self.ffn(self.norm2(x))
        return x
