"""TargetGuidedActivation: bilinear projection between candidate and memory.

Paper Equation 6-7: given target item embedding e_i and memory slots M_k,
activation = softmax(e_i^T W M_k / sqrt(d))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetGuidedActivation(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W.unsqueeze(0))

    def forward(
        self,
        target_emb: torch.Tensor,
        memory_slots: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple:
        """Compute target-guided activation weights over memory slots.

        Args:
            target_emb: (B, d) target item embeddings (positive + negatives)
            memory_slots: (K, d) memory slot matrix
            mask: (B, K) optional boolean mask

        Returns:
            activation: (B, K) attention weights over slots per sample
            scores: (B, K) raw bilinear scores
        """
        K, d = memory_slots.shape
        B = target_emb.shape[0]

        transformed = torch.matmul(memory_slots, self.W)  # (K, d)
        scores = torch.einsum("bd,kd->bk", target_emb, transformed) / (d ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        activation = F.softmax(scores, dim=-1)
        return activation, scores
