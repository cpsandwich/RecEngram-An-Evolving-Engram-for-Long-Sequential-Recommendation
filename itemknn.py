"""ItemKNN: Item-based collaborative filtering (Sarwar et al., 2001).

Uses sparse co-occurrence similarity with top-K neighbor storage.
"""

import torch
import torch.nn as nn
from collections import Counter


class _SparseItemSim:
    """Sparse item-item Jaccard-like similarity with top-K neighbors."""

    def __init__(self, num_neighbors: int = 50):
        self.num_neighbors = num_neighbors
        self.neighbors = {}   # item -> [(neighbor, score), ...]

    @torch.no_grad()
    def build(self, train_seqs: dict, item_vocab_size: int):
        cooccur = Counter()
        item_freq = Counter()

        for seq in train_seqs.values():
            unique = list(set(seq))
            item_freq.update(unique)
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    a, b = unique[i], unique[j]
                    if 0 <= a < item_vocab_size and 0 <= b < item_vocab_size:
                        cooccur[(a, b)] += 1
                        cooccur[(b, a)] += 1

        item_scores = {}
        for (a, b), cnt in cooccur.items():
            denom = item_freq[a] + item_freq[b] - cnt + 1e-8
            sim = cnt / denom
            item_scores.setdefault(a, []).append((b, sim))

        for item, scores_list in item_scores.items():
            scores_list.sort(key=lambda x: -x[1])
            self.neighbors[item] = scores_list[:self.num_neighbors]

    def score(self, item_idx: int, candidate_idx: int) -> float:
        for nbr, sim in self.neighbors.get(item_idx, []):
            if nbr == candidate_idx:
                return sim
        return 0.0


class ItemKNN(nn.Module):
    def __init__(self, item_vocab_size: int, **kwargs):
        super().__init__()
        self.item_vocab_size = item_vocab_size
        self.item_sim = _SparseItemSim()
        self.dummy = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def fit(self, train_seqs: dict):
        self.item_sim.build(train_seqs, self.item_vocab_size)

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]       # (B, L)
        mask = batch["mask"]     # (B, L)
        target = batch["target"]  # (B,)
        negs = batch["negatives"] # (B, N)
        B, L = seq.shape
        N = negs.shape[1]

        t_scores = torch.zeros(B, device=seq.device)
        n_scores = torch.zeros(B, N, device=seq.device)

        for b in range(B):
            valid = seq[b][mask[b]]
            if len(valid) == 0:
                continue
            t_score = sum(self.item_sim.score(i.item(), target[b].item()) for i in valid)
            t_scores[b] = t_score / len(valid)
            for ni in range(N):
                ns = sum(self.item_sim.score(i.item(), negs[b, ni].item()) for i in valid)
                n_scores[b, ni] = ns / len(valid)

        return torch.cat([t_scores.unsqueeze(-1), n_scores], dim=-1)
