"""Tables 5-8: Scenario analyses.

Table 5: Sequence length groups (short/medium/long)
Table 6: Concept drift intensity (3 groups)
Table 7: Interaction density (sparse/medium/dense)
Table 8: Cross-category generalization (seen/unseen categories)
"""

import json
import torch
from pathlib import Path
from collections import defaultdict
from ..config import RecEngramConfig
from ..data.loaders import DATASET_LOADERS
from ..data.preprocessing import (
    preprocess_interactions, chrono_split, build_vocabs,
    compute_drift_intensity, compute_user_drift,
)
from ..data.dataset import SeqRecDataset, collate_fn
from ..models.recengram.model import RecEngram
from ..training.recengram_trainer import run_training_recengram
from ..evaluation.metrics import compute_all_metrics

SCENARIO_MODELS = ["sasrec", "tisasrec", "hpmn", "recengram"]


def run_sequence_length_experiment(
    config: RecEngramConfig,
    dataset: str = "taobao",
    output_dir: str = "results/",
):
    """Table 5: Performance across sequence length groups."""
    output_dir = Path(output_dir) / "table5"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_fn = DATASET_LOADERS[dataset]
    interactions = loader_fn(config)
    config_k = 10 if dataset == "movielens" else 5
    filtered = preprocess_interactions(interactions, k_core=config_k)
    train_data, val_data, test_data = chrono_split(filtered, seed=42)
    vocabs = build_vocabs(train_data)
    item_vocab_size = len(vocabs["item"])
    config.item_vocab_size = item_vocab_size

    seq_lengths = [len(d["items"]) for d in test_data]
    p33 = sorted(seq_lengths)[len(seq_lengths) // 3]
    p67 = sorted(seq_lengths)[2 * len(seq_lengths) // 3]

    def split_by_length(data):
        short = [d for d in data if len(d["items"]) <= p33]
        medium = [d for d in data if p33 < len(d["items"]) <= p67]
        long = [d for d in data if len(d["items"]) > p67]
        return short, medium, long

    results = {}
    groups = {"short": None, "medium": None, "long": None}

    for group_name, group_data in [("short", None), ("medium", None), ("long", None)]:
        short_d, medium_d, long_d = split_by_length(test_data)
        group_data = {"short": short_d, "medium": medium_d, "long": long_d}[group_name]

    for model_name in SCENARIO_MODELS:
        results[model_name] = {}
        torch.manual_seed(42)

        train_ds = SeqRecDataset(train_data, vocabs, config, config.num_negatives)
        val_ds = SeqRecDataset(val_data, vocabs, config, config.num_negatives)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
        )

        if model_name == "recengram":
            model = RecEngram(item_vocab_size=item_vocab_size, **config.__dict__)
            run_training_recengram(
                model, train_loader, val_loader, None, config, verbose=False,
            )
        else:
            from ..models import build_model
            from ..training.trainer import run_training
            model = build_model(model_name, item_vocab_size)
            run_training(
                model, train_loader, val_loader, None, config, model_name=model_name, verbose=False,
            )

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model.eval()

        short_d, medium_d, long_d = split_by_length(test_data)
        for group_name, group_data in [("short", short_d), ("medium", medium_d), ("long", long_d)]:
            if not group_data:
                continue
            ds = SeqRecDataset(group_data, vocabs, config, config.num_negatives)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
            )
            recalls, ndcgs, mrrs = [], [], []
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if model_name == "recengram":
                    scores, _ = model(batch)
                else:
                    scores = model(batch)
                m = compute_all_metrics(scores, k=config.top_k)
                recalls.append(m[f"Recall@{config.top_k}"])
                ndcgs.append(m[f"NDCG@{config.top_k}"])
                mrrs.append(m["MRR"])
            results[model_name][group_name] = {
                "recall": torch.tensor(recalls).mean().item(),
                "ndcg": torch.tensor(ndcgs).mean().item(),
                "mrr": torch.tensor(mrrs).mean().item(),
            }

    out_path = output_dir / f"{dataset}_seq_length.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_concept_drift_experiment(
    config: RecEngramConfig,
    dataset: str = "taobao",
    output_dir: str = "results/",
):
    """Table 6: Performance under different concept drift intensities."""
    output_dir = Path(output_dir) / "table6"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_fn = DATASET_LOADERS[dataset]
    interactions = loader_fn(config)
    config_k = 10 if dataset == "movielens" else 5
    filtered = preprocess_interactions(interactions, k_core=config_k)
    train_data, val_data, test_data = chrono_split(filtered, seed=42)
    vocabs = build_vocabs(train_data)
    item_vocab_size = len(vocabs["item"])
    config.item_vocab_size = item_vocab_size

    user_drifts = compute_user_drift(test_data)
    drifts = list(user_drifts.values())
    if drifts:
        p33 = sorted(drifts)[len(drifts) // 3]
        p67 = sorted(drifts)[2 * len(drifts) // 3]
    else:
        p33, p67 = 0.1, 0.3

    groups = defaultdict(list)
    for d in test_data:
        drift = user_drifts.get(d["user"], 0)
        if drift <= p33:
            groups["low_drift"].append(d)
        elif drift <= p67:
            groups["medium_drift"].append(d)
        else:
            groups["high_drift"].append(d)

    results = {}

    for model_name in SCENARIO_MODELS:
        results[model_name] = {}
        torch.manual_seed(42)

        train_ds = SeqRecDataset(train_data, vocabs, config, config.num_negatives)
        val_ds = SeqRecDataset(val_data, vocabs, config, config.num_negatives)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
        )

        if model_name == "recengram":
            model = RecEngram(item_vocab_size=item_vocab_size, **config.__dict__)
            run_training_recengram(
                model, train_loader, val_loader, None, config, verbose=False,
            )
        else:
            from ..models import build_model
            from ..training.trainer import run_training
            model = build_model(model_name, item_vocab_size)
            run_training(
                model, train_loader, val_loader, None, config, model_name=model_name, verbose=False,
            )

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model.eval()

        for group_name, group_data in groups.items():
            if not group_data:
                continue
            ds = SeqRecDataset(group_data, vocabs, config, config.num_negatives)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
            )
            recalls, ndcgs, mrrs = [], [], []
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if model_name == "recengram":
                    scores, _ = model(batch)
                else:
                    scores = model(batch)
                m = compute_all_metrics(scores, k=config.top_k)
                recalls.append(m[f"Recall@{config.top_k}"])
                ndcgs.append(m[f"NDCG@{config.top_k}"])
                mrrs.append(m["MRR"])
            results[model_name][group_name] = {
                "recall": torch.tensor(recalls).mean().item(),
                "ndcg": torch.tensor(ndcgs).mean().item(),
                "mrr": torch.tensor(mrrs).mean().item(),
            }

    out_path = output_dir / f"{dataset}_drift.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_density_experiment(
    config: RecEngramConfig,
    dataset: str = "taobao",
    output_dir: str = "results/",
):
    """Table 7: Performance under different interaction densities."""
    output_dir = Path(output_dir) / "table7"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_fn = DATASET_LOADERS[dataset]
    interactions = loader_fn(config)
    config_k = 10 if dataset == "movielens" else 5
    filtered = preprocess_interactions(interactions, k_core=config_k)
    train_data, val_data, test_data = chrono_split(filtered, seed=42)
    vocabs = build_vocabs(train_data)
    item_vocab_size = len(vocabs["item"])
    config.item_vocab_size = item_vocab_size

    user_lengths = defaultdict(list)
    for d in test_data:
        user_lengths[d["user"]].append(len(d["items"]))
    avg_lengths = {u: sum(ls) / len(ls) for u, ls in user_lengths.items()}
    values = list(avg_lengths.values())
    if values:
        p33 = sorted(values)[len(values) // 3]
        p67 = sorted(values)[2 * len(values) // 3]
    else:
        p33, p67 = 10, 30

    def split_by_density(data):
        sparse = [d for d in data if avg_lengths.get(d["user"], 0) <= p33]
        medium = [d for d in data if p33 < avg_lengths.get(d["user"], 0) <= p67]
        dense = [d for d in data if avg_lengths.get(d["user"], 0) > p67]
        return sparse, medium, dense

    results = {}

    for model_name in SCENARIO_MODELS:
        results[model_name] = {}
        torch.manual_seed(42)

        train_ds = SeqRecDataset(train_data, vocabs, config, config.num_negatives)
        val_ds = SeqRecDataset(val_data, vocabs, config, config.num_negatives)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
        )

        if model_name == "recengram":
            model = RecEngram(item_vocab_size=item_vocab_size, **config.__dict__)
            run_training_recengram(
                model, train_loader, val_loader, None, config, verbose=False,
            )
        else:
            from ..models import build_model
            from ..training.trainer import run_training
            model = build_model(model_name, item_vocab_size)
            run_training(
                model, train_loader, val_loader, None, config, model_name=model_name, verbose=False,
            )

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model.eval()

        sparse_d, medium_d, dense_d = split_by_density(test_data)
        for group_name, group_data in [("sparse", sparse_d), ("medium", medium_d), ("dense", dense_d)]:
            if not group_data:
                continue
            ds = SeqRecDataset(group_data, vocabs, config, config.num_negatives)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
            )
            recalls, ndcgs, mrrs = [], [], []
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if model_name == "recengram":
                    scores, _ = model(batch)
                else:
                    scores = model(batch)
                m = compute_all_metrics(scores, k=config.top_k)
                recalls.append(m[f"Recall@{config.top_k}"])
                ndcgs.append(m[f"NDCG@{config.top_k}"])
                mrrs.append(m["MRR"])
            results[model_name][group_name] = {
                "recall": torch.tensor(recalls).mean().item(),
                "ndcg": torch.tensor(ndcgs).mean().item(),
                "mrr": torch.tensor(mrrs).mean().item(),
            }

    out_path = output_dir / f"{dataset}_density.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_cross_category_experiment(
    config: RecEngramConfig,
    dataset: str = "taobao",
    output_dir: str = "results/",
):
    """Table 8: Cross-category generalization."""
    output_dir = Path(output_dir) / "table8"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_fn = DATASET_LOADERS[dataset]
    interactions = loader_fn(config)
    config_k = 10 if dataset == "movielens" else 5
    filtered = preprocess_interactions(interactions, k_core=config_k)
    train_data, val_data, test_data = chrono_split(filtered, seed=42)
    vocabs = build_vocabs(train_data)
    item_vocab_size = len(vocabs["item"])
    config.item_vocab_size = item_vocab_size

    seen_test = []
    unseen_test = []
    train_cats = set()
    for d in train_data:
        train_cats.update(d.get("categories", []))

    for d in test_data:
        if d.get("target_category") in train_cats:
            seen_test.append(d)
        else:
            unseen_test.append(d)

    results = {}

    for model_name in SCENARIO_MODELS:
        results[model_name] = {}
        torch.manual_seed(42)

        train_ds = SeqRecDataset(train_data, vocabs, config, config.num_negatives)
        val_ds = SeqRecDataset(val_data, vocabs, config, config.num_negatives)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
        )

        if model_name == "recengram":
            model = RecEngram(item_vocab_size=item_vocab_size, **config.__dict__)
            run_training_recengram(
                model, train_loader, val_loader, None, config, verbose=False,
            )
        else:
            from ..models import build_model
            from ..training.trainer import run_training
            model = build_model(model_name, item_vocab_size)
            run_training(
                model, train_loader, val_loader, None, config, model_name=model_name, verbose=False,
            )

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model.eval()

        for group_name, group_data in [("seen", seen_test), ("unseen", unseen_test)]:
            if not group_data:
                continue
            ds = SeqRecDataset(group_data, vocabs, config, config.num_negatives)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
            )
            recalls, ndcgs, mrrs = [], [], []
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if model_name == "recengram":
                    scores, _ = model(batch)
                else:
                    scores = model(batch)
                m = compute_all_metrics(scores, k=config.top_k)
                recalls.append(m[f"Recall@{config.top_k}"])
                ndcgs.append(m[f"NDCG@{config.top_k}"])
                mrrs.append(m["MRR"])
            results[model_name][group_name] = {
                "recall": torch.tensor(recalls).mean().item(),
                "ndcg": torch.tensor(ndcgs).mean().item(),
                "mrr": torch.tensor(mrrs).mean().item(),
            }

    out_path = output_dir / f"{dataset}_cross_category.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
