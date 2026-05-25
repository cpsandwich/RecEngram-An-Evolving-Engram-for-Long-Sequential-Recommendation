"""BERT4Rec: Bidirectional Encoder for SeqRec (Sun et al., CIKM 2019).

Bidirectional Transformer, no causal mask, last position for prediction.
"""

import torch
import torch.nn as nn
from .common import PositionalEncoding, TransformerBlock


class BERT4Rec(nn.Module):
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

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]
        mask = batch["mask"]
        B, L = seq.shape

        x = self.emb_dropout(self.emb_norm(self.pos_enc(self.item_emb(seq))))

        # Bidirectional: padding mask only
        key_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, L, L)

        for block in self.blocks:
            x = block(x, key_mask)

        lengths = mask.sum(dim=-1).clamp(min=1) - 1
        h_user = x[torch.arange(B, device=x.device), lengths]

        pos = self.item_emb(batch["target"])
        neg = self.item_emb(batch["negatives"])
        pos_scores = (h_user * pos).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg)
        return torch.cat([pos_scores, neg_scores], dim=-1)
