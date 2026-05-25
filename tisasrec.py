"""TiSASRec: Time-Interval Aware Self-Attention (Li et al., WSDM 2020).

Extends SASRec with time-interval-modulated attention weights.
Paper Eq: alpha_ij = softmax(q_i^T k_j / sqrt(d) - gamma_h * |t_i - t_j|)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import PositionalEncoding, FeedForward


class TimeIntervalAttention(nn.Module):
    """Multi-head self-attention with learnable temporal decay per head."""

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
        self.gamma = nn.Parameter(torch.full((num_heads,), 0.01))

    def forward(
        self, x: torch.Tensor, time_gaps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """x: (B, L, d), time_gaps: (B, L, L) |t_i - t_j|, mask: (B, L) bool"""
        B, L, d = x.shape

        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / self.scale  # (B, H, L, L)

        # Time decay: -gamma_h * |t_i - t_j|
        gamma = torch.sigmoid(self.gamma)  # ensure in (0, 1)
        attn = attn - gamma.view(1, -1, 1, 1) * time_gaps.unsqueeze(1)

        # Causal mask
        causal = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~causal, float("-inf"))

        # Padding mask
        pad_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        attn = attn.masked_fill(~pad_mask, float("-inf"))

        attn_w = self.dropout(F.softmax(attn, dim=-1))
        out = torch.einsum("bhqk,bhkd->bhqd", attn_w, v)
        out = out.transpose(1, 2).contiguous().view(B, L, d)
        return self.W_o(out)


class TiSASRec(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(hidden_dim, max_seq_len)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(hidden_dim)

        self.attn_layers = nn.ModuleList([
            TimeIntervalAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            FeedForward(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def _build_time_gaps(self, batch: dict, L: int) -> torch.Tensor:
        """Build time gap matrix (B, L, L) from real timestamps or positions."""
        if "seq_time" in batch and batch["seq_time"] is not None:
            ts = batch["seq_time"].float()  # (B, L)
            gaps = (ts.unsqueeze(-1) - ts.unsqueeze(1)).abs()
            gmax = gaps.max()
            if gmax > 0:
                gaps = gaps / gmax * L  # normalize to sequence range
        else:
            positions = torch.arange(L, device=self.emb_norm.weight.device)
            gaps = (positions.unsqueeze(-1) - positions.unsqueeze(1)).abs().float()
            gaps = gaps.unsqueeze(0).expand(batch["seq"].shape[0], -1, -1)
        return gaps

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]
        mask = batch["mask"]
        B, L = seq.shape

        x = self.emb_dropout(self.emb_norm(self.pos_enc(self.item_emb(seq))))
        time_gaps = self._build_time_gaps(batch, L)

        for i, (attn, ffn) in enumerate(zip(self.attn_layers, self.ffn_layers)):
            x = self.norms[i * 2](x + attn(x, time_gaps, mask))
            x = self.norms[i * 2 + 1](x + ffn(x))

        lengths = mask.sum(dim=-1).clamp(min=1) - 1
        h_user = x[torch.arange(B, device=x.device), lengths]

        pos = self.item_emb(batch["target"])
        neg = self.item_emb(batch["negatives"])
        pos_scores = (h_user * pos).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg)
        return torch.cat([pos_scores, neg_scores], dim=-1)
