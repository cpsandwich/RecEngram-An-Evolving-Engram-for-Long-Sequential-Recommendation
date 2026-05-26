"""Table 2: Overall performance — 10 main models × 3 datasets × 5 seeds."""

import json
import torch
from pathlib import Path
from ..config import RecEngramConfig
from ..data.loaders import DATASET_LOADERS
from ..data.preprocessing import preprocess_interactions, chrono_split, build_vocabs
from ..data.dataset import SeqRecDataset, collate_fn
from ..models import build_model
from ..training.trainer import run_training
from ..training.recengram_trainer import run_training_recengram

MAIN_MODELS = [
    "pop", "itemknn", "gru4rec", "sasrec", "bert4rec",
    "tisasrec", "hpmn", "cl4srec", "engram", "recengram",
]

DATASETS = ["taobao", "amazon_beauty", "movielens"]
DEFAULT_SEEDS = [42, 123, 256, 512, 1024]


def run_overall_experiment(
    config: RecEngramConfig,
    dataset: str = "taobao",
    models: list = None,
    seeds: list = None,
    output_dir: str = "results/",
    checkpoint_dir: str = None,
):
    if models is None:
        models = MAIN_MODELS
    if seeds is None:
        seeds = DEFAULT_SEEDS

    output_dir = Path(output_dir) / "table2"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_fn = DATASET_LOADERS[dataset]
    interactions = loader_fn(config)

    config_k = 10 if dataset == "movielens" else 5
    filtered = preprocess_interactions(interactions, k_core=config_k)

    train_data, val_data, test_data = chrono_split(filtered, seed=42)
    vocabs = build_vocabs(train_data)
    item_vocab_size = len(vocabs["item"])

    config.item_vocab_size = item_vocab_size
    config.category_vocab_size = len(vocabs.get("category", {}))
    config.price_vocab_size = len(vocabs.get("price_tier", {}))
    config.brand_vocab_size = len(vocabs.get("brand", {}))

    results = {}

    for model_name in models:
        model_results = {}
        print(f"\n{'='*60}")
        print(f"Model: {model_name} | Dataset: {dataset}")
        print(f"{'='*60}")

        for seed in seeds:
            torch.manual_seed(seed)
            print(f"\n  Seed {seed}...")

            train_ds = SeqRecDataset(train_data, vocabs, config, config.num_negatives)
            val_ds = SeqRecDataset(val_data, vocabs, config, config.num_negatives)
            test_ds = SeqRecDataset(test_data, vocabs, config, config.num_negatives)

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=config.batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=config.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True,
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=config.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True,
            )

            model = build_model(model_name, item_vocab_size)

            if model_name in ("pop", "itemknn"):
                model.fit(train_data, vocabs)
                from ..training.trainer import evaluate
                test_metrics = evaluate(model, test_loader, config.device, config.top_k)
                result = {"test_metrics": test_metrics}
            elif model_name in ("recengram", "engram"):
                result = run_training_recengram(
                    model, train_loader, val_loader, test_loader, config,
                    verbose=False,
                )
            else:
                result = run_training(
                    model, train_loader, val_loader, test_loader, config,
                    model_name=model_name, verbose=False,
                )

            model_results[f"seed_{seed}"] = {
                "recall": result["test_metrics"].get("recall", 0),
                "ndcg": result["test_metrics"].get("ndcg", 0),
                "hr": result["test_metrics"].get("hr", 0),
                "mrr": result["test_metrics"].get("mrr", 0),
            }

            print(f"    Recall@{config.top_k}: {model_results[f'seed_{seed}']['recall']:.4f}")
            print(f"    NDCG@{config.top_k}: {model_results[f'seed_{seed}']['ndcg']:.4f}")

        recalls = [v["recall"] for v in model_results.values()]
        ndcgs = [v["ndcg"] for v in model_results.values()]
        hrs = [v["hr"] for v in model_results.values()]
        mrrs = [v["mrr"] for v in model_results.values()]

        results[model_name] = {
            "recall_mean": sum(recalls) / len(recalls),
            "recall_std": (sum((r - sum(recalls)/len(recalls))**2 for r in recalls) / len(recalls)) ** 0.5,
            "ndcg_mean": sum(ndcgs) / len(ndcgs),
            "ndcg_std": (sum((r - sum(ndcgs)/len(ndcgs))**2 for r in ndcgs) / len(ndcgs)) ** 0.5,
            "hr_mean": sum(hrs) / len(hrs),
            "hr_std": (sum((r - sum(hrs)/len(hrs))**2 for r in hrs) / len(hrs)) ** 0.5,
            "mrr_mean": sum(mrrs) / len(mrrs),
            "mrr_std": (sum((r - sum(mrrs)/len(mrrs))**2 for r in mrrs) / len(mrrs)) ** 0.5,
        }

    out_path = output_dir / f"{dataset}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    return results
