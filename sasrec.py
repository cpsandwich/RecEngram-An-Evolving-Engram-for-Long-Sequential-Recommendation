"""SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018).

Causal Transformer decoder: predicts next item from previous items.
"""

import torch
import torch.nn as nn
from .common import PositionalEncoding, TransformerBlock


class SASRec(nn.Module):
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
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular causal mask (L, L)."""
        return torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]        # (B, L)
        mask = batch["mask"]      # (B, L)
        B, L = seq.shape

        x = self.emb_dropout(self.emb_norm(self.pos_enc(self.item_emb(seq))))

        causal = self._causal_mask(L, x.device)
        # Combine causal + padding masks
        key_mask = mask.unsqueeze(1)  # (B, 1, L)
        attn_mask = causal.unsqueeze(0) & key_mask  # (B, L, L)

        for block in self.blocks:
            x = block(x, attn_mask)

        # Last valid position
        lengths = mask.sum(dim=-1).clamp(min=1) - 1
        h_user = x[torch.arange(B, device=x.device), lengths]  # (B, d)

        pos = self.item_emb(batch["target"])
        neg = self.item_emb(batch["negatives"])
        pos_scores = (h_user * pos).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg)
        return torch.cat([pos_scores, neg_scores], dim=-1)
