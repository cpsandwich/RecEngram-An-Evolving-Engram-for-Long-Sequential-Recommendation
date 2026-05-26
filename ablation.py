"""Table 4: Ablation study — additive, subtractive, slot sensitivity."""

import json
import torch
from pathlib import Path
from ..config import RecEngramConfig
from ..data.loaders import DATASET_LOADERS
from ..data.preprocessing import preprocess_interactions, chrono_split, build_vocabs
from ..data.dataset import SeqRecDataset, collate_fn
from ..models import build_model
from ..models.recengram.model import RecEngram
from ..training.recengram_trainer import run_training_recengram


def run_ablation(
    config: RecEngramConfig,
    dataset: str = "taobao",
    experiment: str = "additive",
    output_dir: str = "results/",
):
    """Run ablation experiments.

    Args:
        experiment: "additive", "subtractive", or "slots"
    """
    output_dir = Path(output_dir) / "table4"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_fn = DATASET_LOADERS[dataset]
    interactions = loader_fn(config)

    config_k = 10 if dataset == "movielens" else 5
    filtered = preprocess_interactions(interactions, k_core=config_k)
    train_data, val_data, test_data = chrono_split(filtered, seed=42)
    vocabs = build_vocabs(train_data)
    item_vocab_size = len(vocabs["item"])
    config.item_vocab_size = item_vocab_size

    torch.manual_seed(42)
    train_ds = SeqRecDataset(train_data, vocabs, config, config.num_negatives)
    val_ds = SeqRecDataset(val_data, vocabs, config, config.num_negatives)
    test_ds = SeqRecDataset(test_data, vocabs, config, config.num_negatives)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=config.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers,
    )

    results = {}

    if experiment == "additive":
        variants = {
            "sasrec_base": {"model": "sasrec", "recengram": False},
            "+memory": {
                "recengram": True,
                "use_target_guided": False,
                "use_surprise": False,
                "use_hierarchy": False,
            },
            "+target_guided": {
                "recengram": True,
                "use_target_guided": True,
                "use_surprise": False,
                "use_hierarchy": False,
            },
            "+surprise": {
                "recengram": True,
                "use_target_guided": True,
                "use_surprise": True,
                "use_hierarchy": False,
            },
            "+hierarchy (full)": {
                "recengram": True,
                "use_target_guided": True,
                "use_surprise": True,
                "use_hierarchy": True,
            },
        }
    elif experiment == "subtractive":
        variants = {
            "full_recengram": {
                "recengram": True,
                "use_target_guided": True,
                "use_surprise": True,
                "use_hierarchy": True,
            },
            "-target_guided": {
                "recengram": True,
                "use_target_guided": False,
                "use_surprise": True,
                "use_hierarchy": True,
            },
            "-surprise": {
                "recengram": True,
                "use_target_guided": True,
                "use_surprise": False,
                "use_hierarchy": True,
            },
            "-hierarchy": {
                "recengram": True,
                "use_target_guided": True,
                "use_surprise": True,
                "use_hierarchy": False,
            },
        }
    elif experiment == "slots":
        variants = {}
        for num_slots in [32, 64, 96, 128, 160, 192, 256]:
            long_slots = num_slots // 4
            short_slots = num_slots - long_slots
            variants[f"K={num_slots}"] = {
                "recengram": True,
                "use_target_guided": True,
                "use_surprise": True,
                "use_hierarchy": True,
                "num_memory_slots": num_slots,
                "num_long_slots": long_slots,
                "num_short_slots": short_slots,
            }
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    for variant_name, kwargs in variants.items():
        print(f"\n--- {variant_name} ---")
        torch.manual_seed(42)

        if kwargs.pop("recengram", False):
            model = RecEngram(item_vocab_size=item_vocab_size, **{**config.__dict__, **kwargs})
            result = run_training_recengram(
                model, train_loader, val_loader, test_loader, config, verbose=False,
            )
        else:
            model = build_model(kwargs.pop("model"), item_vocab_size)
            from ..training.trainer import run_training
            result = run_training(
                model, train_loader, val_loader, test_loader, config,
                model_name=variant_name, verbose=False,
            )

        results[variant_name] = {
            "recall": result["test_metrics"].get("recall", 0),
            "ndcg": result["test_metrics"].get("ndcg", 0),
            "hr": result["test_metrics"].get("hr", 0),
            "mrr": result["test_metrics"].get("mrr", 0),
        }
        print(f"  Recall@10: {results[variant_name]['recall']:.4f}")
        print(f"  NDCG@10: {results[variant_name]['ndcg']:.4f}")

    out_path = output_dir / f"{experiment}_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results
