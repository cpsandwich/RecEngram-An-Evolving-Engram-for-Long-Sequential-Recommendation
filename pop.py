"""POP: Popularity-based non-personalized recommendation baseline.

Ranks items by global frequency. Lower bound for all sequential methods.
"""

import torch
import torch.nn as nn
from collections import Counter


class POP(nn.Module):
    def __init__(self, item_vocab_size: int, **kwargs):
        super().__init__()
        self.register_buffer("pop_scores", torch.zeros(item_vocab_size))
        # dummy param so model.parameters() isn't empty
        self.dummy = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def fit(self, train_seqs: dict):
        """Compute item frequencies from training sequences."""
        counter = Counter()
        for seq in train_seqs.values():
            for idx in seq:
                if 0 <= idx < len(self.pop_scores):
                    counter[idx] += 1
        total = max(sum(counter.values()), 1)
        for idx, cnt in counter.items():
            self.pop_scores[idx] = cnt / total

    def forward(self, batch: dict) -> torch.Tensor:
        target = self.pop_scores[batch["target"]]       # (B,)
        negs = self.pop_scores[batch["negatives"]]      # (B, N)
        return torch.cat([target.unsqueeze(-1), negs], dim=-1)
