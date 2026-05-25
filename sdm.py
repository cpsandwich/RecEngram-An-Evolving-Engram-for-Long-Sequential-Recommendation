"""SDM: Session-based Deep Matching (Lv et al., CIKM 2019).

Multi-head self-attention over recent session + user embedding gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MultiHeadAttention


class UserEncoder(nn.Module):
    """Encode long-term user profile from all behavior embeddings."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, seq_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=-1).clamp(min=1).unsqueeze(-1).float()
        user_emb = seq_emb.sum(dim=1) / lengths
        return self.fc(user_emb)


class InterestFusion(nn.Module):
    """Gated fusion of short-term and long-term interests."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, short_term: torch.Tensor, long_term: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([short_term, long_term], dim=-1))
        return g * short_term + (1 - g) * long_term


class SDM(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        num_heads: int = 2,
        dropout: float = 0.2,
        short_session_len: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.short_session_len = short_session_len

        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.multi_head_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.user_encoder = UserEncoder(hidden_dim)
        self.fusion = InterestFusion(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]
        mask = batch["mask"]
        B, L = seq.shape

        seq_emb = self.emb_dropout(self.item_emb(seq))

        short_mask = mask.clone()
        short_mask[:, :-self.short_session_len] = False if self.short_session_len < L else True

        if short_mask.any():
            attn_mask = short_mask.unsqueeze(1) & short_mask.unsqueeze(2)
            short_out = self.multi_head_attn(seq_emb, seq_emb, seq_emb, ~attn_mask)
            short_lengths = short_mask.sum(dim=-1).clamp(min=1).unsqueeze(-1)
            short_interest = short_out.sum(dim=1) / short_lengths.float()
        else:
            short_interest = torch.zeros(B, self.hidden_dim, device=seq.device)

        user_emb = self.user_encoder(seq_emb, mask)

        h_user = self.fusion(short_interest, user_emb)

        pos = self.item_emb(batch["target"])
        neg = self.item_emb(batch["negatives"])
        pos_scores = (h_user * pos).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg)
        return torch.cat([pos_scores, neg_scores], dim=-1)
