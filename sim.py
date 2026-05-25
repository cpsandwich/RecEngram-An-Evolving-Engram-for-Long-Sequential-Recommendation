"""SIM: Search-based Interest Model (Pi et al., CIKM 2020).

Two-stage retrieval for long user behavior sequences:
- GSU (General Search Unit): select top-K behaviors via dot-product search
- ESU (Exact Search Unit): exact multi-head attention over selected behaviors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MultiHeadAttention


class GeneralSearchUnit(nn.Module):
    """Hard search: select top-K behaviors with highest dot-product similarity."""

    def __init__(self, top_k: int = 50):
        super().__init__()
        self.top_k = top_k

    def forward(
        self,
        target_emb: torch.Tensor,
        seq_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple:
        B, L, d = seq_emb.shape
        top_k = min(self.top_k, L)

        sim = torch.einsum("bd,bld->bl", target_emb, seq_emb)
        sim = sim.masked_fill(~mask, float("-inf"))

        _, top_idx = sim.topk(top_k, dim=-1)

        selected = torch.gather(
            seq_emb, 1,
            top_idx.unsqueeze(-1).expand(-1, -1, d)
        )

        selected_mask = torch.gather(mask, 1, top_idx)

        return selected, selected_mask, top_idx


class ExactSearchUnit(nn.Module):
    """Exact attention over selected behaviors."""

    def __init__(self, hidden_dim: int, num_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, target: torch.Tensor, selected: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target_expanded = target.unsqueeze(1)
        out = self.attn(target_expanded, selected, selected, ~mask.unsqueeze(1))
        return self.norm(out.squeeze(1))


class SIM(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        num_heads: int = 2,
        dropout: float = 0.2,
        gsu_top_k: int = 50,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.gsu = GeneralSearchUnit(gsu_top_k)
        self.esu = ExactSearchUnit(hidden_dim, num_heads, dropout)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]
        mask = batch["mask"]
        B, L = seq.shape

        seq_emb = self.emb_dropout(self.item_emb(seq))
        target_emb = self.item_emb(batch["target"])
        neg_emb = self.item_emb(batch["negatives"])

        selected, sel_mask, _ = self.gsu(target_emb, seq_emb, mask)
        h_user = self.esu(target_emb, selected, sel_mask)

        pos_scores = (h_user * target_emb).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg_emb)
        return torch.cat([pos_scores, neg_scores], dim=-1)
