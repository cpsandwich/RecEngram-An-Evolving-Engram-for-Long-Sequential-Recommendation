"""PyTorch Dataset and collate function."""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SeqRecDataset(Dataset):
    """Sequential recommendation dataset.

    Produces (input_seq, target, negatives) for each valid training instance.
    Uses popularity-based negative sampling (paper Eq. 11).
    """

    def __init__(
        self,
        user2seq: Dict[str, List[str]],
        seq2attr: Dict[str, dict],
        item2idx: Dict[str, int],
        cat2idx: Dict[str, int],
        price2idx: Dict[str, int],
        brand2idx: Dict[str, int],
        max_seq_len: int = 500,
        num_negatives: int = 8,
        include_timestamps: bool = False,
        user2time: Optional[Dict[str, List[int]]] = None,
    ):
        self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        self.include_timestamps = include_timestamps
        self.item2idx = item2idx
        self.cat2idx = cat2idx
        self.price2idx = price2idx
        self.brand2idx = brand2idx
        self.seq2attr = seq2attr

        # Build samples: (uid, input_seq, target_item, target_time)
        self.samples = []
        self.user_total_inter = {}   # uid -> total interactions for density analysis

        for uid, seq in user2seq.items():
            self.user_total_inter[uid] = len(seq)
            if len(seq) < 2:
                continue
            times = user2time.get(uid, []) if user2time else []
            for i in range(1, len(seq)):
                input_seq = seq[max(0, i - max_seq_len):i]
                target = seq[i]
                target_time = times[i] if i < len(times) else 0
                self.samples.append((uid, input_seq, target, target_time))

        # Build popularity distribution for negative sampling
        self._build_pop_sampler(user2seq)

    def _build_pop_sampler(self, user2seq: Dict):
        counter = Counter()
        for seq in user2seq.values():
            for iid in seq:
                counter[self.item2idx.get(iid, 0)] += 1
        total = sum(counter.values())
        self.pop_items = list(counter.keys())
        self.pop_weights = np.array(
            [(counter[idx] + 1) / (total + len(counter)) for idx in self.pop_items],
            dtype=np.float32,
        )
        self.pop_weights /= self.pop_weights.sum()

    def _get_attr(self, item_str: str) -> dict:
        attr = self.seq2attr.get(item_str, {})
        return {
            "cat": self.cat2idx.get(attr.get("category", 0), 0),
            "price": self.price2idx.get(attr.get("price_tier", 0), 0),
            "brand": self.brand2idx.get(attr.get("brand", 0), 0),
        }

    def _sample_negatives(self) -> np.ndarray:
        return np.random.choice(self.pop_items, size=self.num_negatives,
                                p=self.pop_weights)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, input_seq, target, target_time = self.samples[idx]

        # Pad/truncate input
        seq_ids = [self.item2idx.get(i, 0) for i in input_seq]
        seq_ids = seq_ids[-self.max_seq_len:]
        seq_len = len(seq_ids)
        padded = seq_ids + [0] * (self.max_seq_len - seq_len)
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:seq_len] = True

        # Sequence attributes
        seq_attrs = [self._get_attr(i) for i in input_seq[-self.max_seq_len:]]
        seq_attrs += [{"cat": 0, "price": 0, "brand": 0}] * (self.max_seq_len - seq_len)

        # Target attributes
        target_attr = self._get_attr(target)

        # Negatives
        neg_indices = self._sample_negatives()

        out = {
            "uid": uid,
            "seq": torch.tensor(padded, dtype=torch.long),
            "seq_len": seq_len,
            "mask": mask,
            "seq_cat": torch.tensor([a["cat"] for a in seq_attrs], dtype=torch.long),
            "seq_price": torch.tensor([a["price"] for a in seq_attrs], dtype=torch.long),
            "seq_brand": torch.tensor([a["brand"] for a in seq_attrs], dtype=torch.long),
            "target": torch.tensor(self.item2idx.get(target, 0), dtype=torch.long),
            "target_cat": torch.tensor(target_attr["cat"], dtype=torch.long),
            "target_price": torch.tensor(target_attr["price"], dtype=torch.long),
            "target_brand": torch.tensor(target_attr["brand"], dtype=torch.long),
            "negatives": torch.tensor(neg_indices, dtype=torch.long),
        }

        if self.include_timestamps:
            out["target_time"] = torch.tensor(target_time, dtype=torch.long)

        return out


def collate_fn(batch: List[dict]) -> dict:
    """Custom collate for variable-length sequences."""
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[k] = torch.stack(vals)
        else:
            collated[k] = vals
    return collated
