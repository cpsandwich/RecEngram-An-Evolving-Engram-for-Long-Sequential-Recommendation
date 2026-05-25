"""DIN: Deep Interest Network (Zhou et al., KDD 2018).

Adaptive attention over user behavior sequence given target item.
Uses Dice activation and attention pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice(nn.Module):
    """Data-adaptive activation function from DIN paper."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))
        self.bn = nn.BatchNorm1d(hidden_dim, eps=1e-8, momentum=0.99)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x_norm = self.bn(x)
            p = torch.sigmoid(x_norm)
            return p * x + (1 - p) * self.alpha * x
        return x


class AttentionUnit(nn.Module):
    """Attention unit for computing relevance between target and behavior."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 80),
            Dice(80),
            nn.Linear(80, 40),
            Dice(40),
            nn.Linear(40, 1),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor, keys_len: torch.Tensor) -> torch.Tensor:
        """query: (B, d), keys: (B, L, d), keys_len: (B,)"""
        B, L, d = keys.shape
        query_expand = query.unsqueeze(1).expand(-1, L, -1)
        attn_input = torch.cat([
            query_expand, keys, query_expand - keys, query_expand * keys
        ], dim=-1)
        attn_scores = self.fc(attn_input).squeeze(-1)

        mask = torch.arange(L, device=keys.device).unsqueeze(0) >= keys_len.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        return F.softmax(attn_scores, dim=-1)


class DIN(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.attention = AttentionUnit(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, 200),
            Dice(200),
            nn.Linear(200, 80),
            Dice(80),
            nn.Linear(80, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]
        mask = batch["mask"]
        B, L = seq.shape

        seq_emb = self.emb_dropout(self.item_emb(seq))
        target_emb = self.item_emb(batch["target"])
        neg_emb = self.item_emb(batch["negatives"])

        lengths = mask.sum(dim=-1).clamp(min=1)

        attn_w = self.attention(target_emb, seq_emb, lengths)
        user_interest = torch.einsum("bl,bld->bd", attn_w, seq_emb)

        concat = torch.cat([
            user_interest,
            target_emb,
            user_interest * target_emb,
        ], dim=-1)

        h_user = self.fc(concat)

        pos_scores = (h_user * target_emb).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg_emb)
        return torch.cat([pos_scores, neg_scores], dim=-1)
