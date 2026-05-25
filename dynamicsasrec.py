"""DynamicSASRec: SASRec + learnable temporal decay (Appendix D.3).

Same as TiSASRec but with SASRec architecture + learnable gamma per head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import PositionalEncoding, FeedForward


class DynamicSASRec(nn.Module):
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

        self.blocks = nn.ModuleList([
            _DynamicSASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def _build_time_gaps(self, batch: dict, L: int) -> torch.Tensor:
        if "seq_time" in batch and batch["seq_time"] is not None:
            ts = batch["seq_time"].float()
            gaps = (ts.unsqueeze(-1) - ts.unsqueeze(1)).abs()
            gmax = gaps.max()
            if gmax > 0:
                gaps = gaps / gmax * L
        else:
            positions = torch.arange(L, device=self.emb_norm.weight.device)
            gaps = positions.unsqueeze(-1) - positions.unsqueeze(1)
            gaps = gaps.abs().float().unsqueeze(0).expand(batch["seq"].shape[0], -1, -1)
        return gaps

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]
        mask = batch["mask"]
        B, L = seq.shape

        x = self.emb_dropout(self.emb_norm(self.pos_enc(self.item_emb(seq))))
        time_gaps = self._build_time_gaps(batch, L)

        causal = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        key_mask = mask.unsqueeze(1)
        attn_mask = causal.unsqueeze(0) & key_mask

        for block in self.blocks:
            x = block(x, time_gaps, attn_mask)

        lengths = mask.sum(dim=-1).clamp(min=1) - 1
        h_user = x[torch.arange(B, device=x.device), lengths]

        pos = self.item_emb(batch["target"])
        neg = self.item_emb(batch["negatives"])
        pos_scores = (h_user * pos).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg)
        return torch.cat([pos_scores, neg_scores], dim=-1)


class _DynamicSASRecBlock(nn.Module):
    """SASRec-style block with time-interval decay in attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
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
        self.gamma = nn.Parameter(torch.full((num_heads,), 0.01))
        self.gamma_sigmoid = True

        self.ffn = FeedForward(hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, time_gaps: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        B, L, d = x.shape
        residual = x
        x = self.norm1(x)

        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / self.scale
        gamma = torch.sigmoid(self.gamma) if self.gamma_sigmoid else self.gamma.abs()
        attn = attn - gamma.view(1, -1, 1, 1) * time_gaps.unsqueeze(1)

        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn_w = self.dropout(F.softmax(attn, dim=-1))

        out = torch.einsum("bhqk,bhkd->bhqd", attn_w, v)
        out = out.transpose(1, 2).contiguous().view(B, L, d)

        x = residual + self.dropout(self.W_o(out))
        x = x + self.ffn(self.norm2(x))
        return x
