"""Raw data loaders for Taobao, Amazon Beauty, MovieLens1M."""

import csv
import gzip
import json
import os
from typing import List, Dict


def load_taobao(path: str) -> List[dict]:
    """Taobao User Behavior CSV (uid, iid, cid, brand, timestamp)."""
    data = []
    fname = os.path.join(path, "UserBehavior.csv")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Taobao data not found at {fname}")
    with open(fname, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 5:
                continue
            data.append({
                "user": row[0],
                "item": row[1],
                "category": row[2],
                "brand": row[3],
                "price_tier": 0,
                "time": int(row[4]),
            })
    return data


def load_amazon_beauty(path: str) -> List[dict]:
    """Amazon Beauty 5-core reviews (json.gz)."""
    data = []
    fname = os.path.join(path, "reviews_Beauty_5.json.gz")
    if not os.path.exists(fname):
        # Try uncompressed
        fname = os.path.join(path, "reviews_Beauty_5.json")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Amazon Beauty data not found at {path}")
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname, "r") as f:
        for line in f:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            r = json.loads(line)
            data.append({
                "user": r.get("reviewerID", ""),
                "item": r.get("asin", ""),
                "category": r.get("category", "Beauty"),
                "price_tier": 0,
                "brand": r.get("brand", ""),
                "time": r.get("unixReviewTime", 0),
            })
    return data


def load_movielens(path: str) -> List[dict]:
    """MovieLens1M ratings.dat (uid::iid::rating::timestamp)."""
    data = []
    fname = os.path.join(path, "ratings.dat")
    if not os.path.exists(fname):
        fname = os.path.join(path, "ml-1m", "ratings.dat")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"MovieLens1M data not found at {path}")
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            uid, iid, rating, ts = line.strip().split("::")
            if int(rating) < 4:        # implicit: keep ratings >= 4
                continue
            data.append({
                "user": uid,
                "item": iid,
                "category": 0,
                "price_tier": 0,
                "brand": 0,
                "time": int(ts),
            })
    return data


DATASET_LOADERS = {
    "taobao": load_taobao,
    "amazon_beauty": load_amazon_beauty,
    "movielens": load_movielens,
}
