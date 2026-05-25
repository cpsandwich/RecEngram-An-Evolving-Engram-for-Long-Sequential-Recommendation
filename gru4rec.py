"""GRU4Rec: Session-based RNN for recommendation (Hidasi et al., ICLR 2016).

Multi-layer GRU over item sequence, last hidden state as user representation.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRU4Rec(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        num_layers: int = 3,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]        # (B, L)
        mask = batch["mask"]      # (B, L)
        B, L = seq.shape

        x = self.emb_dropout(self.item_emb(seq))

        # Pack for variable-length sequences
        lengths = mask.sum(dim=-1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)  # hidden: (num_layers, B, d)
        h_user = hidden[-1]           # last layer

        pos = self.item_emb(batch["target"])      # (B, d)
        neg = self.item_emb(batch["negatives"])   # (B, N, d)

        pos_scores = (h_user * pos).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg)
        return torch.cat([pos_scores, neg_scores], dim=-1)
