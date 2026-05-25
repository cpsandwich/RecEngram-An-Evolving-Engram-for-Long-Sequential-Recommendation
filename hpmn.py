"""HPMN: Hierarchical Periodic Memory Network (Ren et al., CIKM 2020).

Multi-hop memory reading over K learned memory slots with temporal gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HPMN(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        memory_slots: int = 64,
        num_hops: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.num_hops = num_hops

        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=1,
            batch_first=True,
        )

        self.memory = nn.Parameter(torch.randn(memory_slots, hidden_dim))

        self.mem_update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.query_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.hop_rnn = nn.GRUCell(hidden_dim, hidden_dim)

        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.memory)

    def _encode_sequence(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode item sequence with GRU, return last valid state."""
        B, L = seq.shape
        x = self.emb_dropout(self.item_emb(seq))
        lengths = mask.sum(dim=-1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    def _multi_hop_read(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """Multi-hop memory reading.

        Args:
            query: (B, d) initial query
            memory: (K, d) memory slots

        Returns:
            (B, d) final readout vector
        """
        B, d = query.shape
        K = memory.shape[0]
        mem_expanded = memory.unsqueeze(0).expand(B, -1, -1)

        q = query
        for hop in range(self.num_hops):
            attn = torch.einsum("bd,bkd->bk", q, mem_expanded) / (d ** 0.5)
            attn_w = F.softmax(attn, dim=-1)
            readout = torch.einsum("bk,bkd->bd", attn_w, mem_expanded)

            q = self.hop_rnn(readout, q)

        return q

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]
        mask = batch["mask"]
        B, L = seq.shape

        h_gru = self._encode_sequence(seq, mask)

        memory_norm = F.normalize(self.memory, dim=-1)
        h_mem = self._multi_hop_read(h_gru, memory_norm)
        h_user = self.fusion(torch.cat([h_gru, h_mem], dim=-1))
        h_user = self.norm(h_user)

        pos = self.item_emb(batch["target"])
        neg = self.item_emb(batch["negatives"])
        pos_scores = (h_user * pos).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg)
        return torch.cat([pos_scores, neg_scores], dim=-1)
