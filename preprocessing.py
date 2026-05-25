"""Preprocessing: 5/10-core filter, chronological split, vocab building."""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


def preprocess_interactions(
    raw_data: List[dict],
    min_user_inter: int = 5,
    min_item_inter: int = 10,
    max_seq_len: int = 500,
) -> Tuple[Dict, Dict, Dict]:
    """Filter, sort, truncate — returns user2seq, seq2attr, stats."""
    # Filter users
    user_counts = Counter(r["user"] for r in raw_data)
    valid_users = {u for u, c in user_counts.items() if c >= min_user_inter}
    filtered = [r for r in raw_data if r["user"] in valid_users]

    # Filter items
    item_counts = Counter(r["item"] for r in filtered)
    valid_items = {i for i, c in item_counts.items() if c >= min_item_inter}
    filtered = [r for r in filtered if r["item"] in valid_items]

    # Build attribute map
    seq2attr = {}
    for r in filtered:
        if r["item"] not in seq2attr:
            seq2attr[r["item"]] = {
                "category": r.get("category", 0),
                "price_tier": r.get("price_tier", 0),
                "brand": r.get("brand", 0),
            }

    # Sort by (user, time) and build sequences
    filtered.sort(key=lambda x: (x["user"], x["time"]))
    user2seq = defaultdict(list)
    for r in filtered:
        user2seq[r["user"]].append(r["item"])

    # Truncate to max_seq_len
    for u in user2seq:
        if len(user2seq[u]) > max_seq_len:
            user2seq[u] = user2seq[u][-max_seq_len:]

    seq_lens = [len(s) for s in user2seq.values()]
    stats = {
        "num_users": len(user2seq),
        "num_items": len(valid_items),
        "num_interactions": sum(seq_lens),
        "avg_length": float(np.mean(seq_lens)),
    }
    return dict(user2seq), seq2attr, stats


def chrono_split(
    user2seq: Dict[str, List[str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Tuple[Dict, Dict, Dict]:
    """Chronological 7:1:2 split per user."""
    train_seqs, val_seqs, test_seqs = {}, {}, {}
    for uid, seq in user2seq.items():
        n = len(seq)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        train_seqs[uid] = seq[:train_end]
        val_seqs[uid] = seq[train_end:val_end]
        test_seqs[uid] = seq[val_end:]
    return train_seqs, val_seqs, test_seqs


def build_vocabs(
    train_seqs: Dict[str, List[str]],
    seq2attr: Dict[str, dict],
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Build item/category/price/brand string-to-int mappings."""
    all_items = set()
    categories = set()
    prices = set()
    brands = set()
    for seq in train_seqs.values():
        for iid in seq:
            all_items.add(iid)
            attr = seq2attr.get(iid, {})
            categories.add(attr.get("category", 0))
            prices.add(attr.get("price_tier", 0))
            brands.add(attr.get("brand", 0))

    item2idx = {i: idx + 1 for idx, i in enumerate(sorted(all_items))}
    cat2idx = {c: idx + 1 for idx, c in enumerate(sorted(categories))}
    price2idx = {p: idx + 1 for idx, p in enumerate(sorted(prices))}
    brand2idx = {b: idx + 1 for idx, b in enumerate(sorted(brands))}

    # Padding idx = 0
    for m in [item2idx, cat2idx, price2idx, brand2idx]:
        m["<PAD>"] = 0
    return item2idx, cat2idx, price2idx, brand2idx


def compute_drift_intensity(user2seq: Dict, seq2attr: Dict) -> float:
    """Dataset-level category switch rate."""
    rates = []
    for seq in user2seq.values():
        if len(seq) < 2:
            continue
        cats = [seq2attr.get(i, {}).get("category", 0) for i in seq]
        switches = sum(1 for a, b in zip(cats, cats[1:]) if a != b)
        rates.append(switches / (len(seq) - 1))
    return float(np.mean(rates)) if rates else 0.0


def compute_user_drift(user_seq: List[str], seq2attr: Dict) -> float:
    """Per-user category switch rate."""
    if len(user_seq) < 2:
        return 0.0
    cats = [seq2attr.get(i, {}).get("category", 0) for i in user_seq]
    switches = sum(1 for a, b in zip(cats, cats[1:]) if a != b)
    return switches / (len(user_seq) - 1)
