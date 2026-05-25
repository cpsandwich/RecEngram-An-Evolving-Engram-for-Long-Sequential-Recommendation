"""MIND: Multi-Interest Network with Dynamic Routing (Li et al., CIKM 2019).

Extracts K interest capsules from user behavior via dynamic routing.
Prediction: max dot product across all interest vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRouting(nn.Module):
    """Capsule routing: iteratively refine coupling coefficients between
    behavior embeddings (low-level capsules) and interest capsules."""

    def __init__(self, hidden_dim: int, num_interests: int = 4, num_iter: int = 3):
        super().__init__()
        self.num_interests = num_interests
        self.num_iter = num_iter
        self.S = nn.Parameter(torch.randn(1, hidden_dim, num_interests))
        nn.init.xavier_uniform_(self.S)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d), mask: (B, L)"""
        B, L, d = x.shape

        S = self.S.expand(B, -1, -1)
        B_ij = torch.zeros(B, L, self.num_interests, device=x.device)

        for _ in range(self.num_iter):
            C_ij = F.softmax(B_ij, dim=-1)
            Z_j = torch.einsum("bld,blk->bkd", x, C_ij)
            V_j = self._squash(Z_j)
            if _ < self.num_iter - 1:
                B_ij = B_ij + torch.einsum("bld,bkd->blk", x, V_j)

        return V_j

    def _squash(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = z.norm(dim=-1, keepdim=True)
        return (z_norm ** 2 / (1 + z_norm ** 2)) * (z / (z_norm + 1e-8))


class MIND(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        dropout: float = 0.2,
        num_interests: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_interests = num_interests

        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.behavior_fc = nn.Linear(hidden_dim, hidden_dim)
        self.capsule = DynamicRouting(hidden_dim, num_interests)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]
        mask = batch["mask"]
        B, L = seq.shape

        seq_emb = self.emb_dropout(self.item_emb(seq))
        seq_emb = self.behavior_fc(seq_emb)

        interests = self.capsule(seq_emb, mask)

        target_emb = self.item_emb(batch["target"])
        neg_emb = self.item_emb(batch["negatives"])

        target_norm = F.normalize(target_emb, dim=-1)
        interests_norm = F.normalize(interests, dim=-1)

        pos_scores = torch.einsum("bd,bkd->bk", target_norm, interests_norm)
        pos_scores = pos_scores.max(dim=-1, keepdim=True).values

        neg_emb_flat = neg_emb.view(-1, self.hidden_dim)
        neg_norm = F.normalize(neg_emb_flat, dim=-1)
        neg_all = torch.einsum("bkd,nd->bnk", interests_norm, neg_norm)
        neg_scores = neg_all.max(dim=-1).values

        return torch.cat([pos_scores, neg_scores], dim=-1)
