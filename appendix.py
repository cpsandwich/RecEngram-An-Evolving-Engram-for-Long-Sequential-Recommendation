"""Appendix experiments.

- Alibaba baseline results (DIN, DIEN, MIND, SDM, SIM) on all 3 datasets
- Standard deviation tables for all models
- DynamicSASRec comparison
"""

import json
import torch
from pathlib import Path
from ..config import RecEngramConfig
from ..data.loaders import DATASET_LOADERS
from ..data.preprocessing import preprocess_interactions, chrono_split, build_vocabs
from ..data.dataset import SeqRecDataset, collate_fn
from ..models import build_model
from ..training.trainer import run_training

ALIBABA_MODELS = ["din", "dien", "mind", "sdm", "sim"]
APPENDIX_MODELS = ["sasrec", "tisasrec", "dynamicsasrec", "recengram"]
DATASETS = ["taobao", "amazon_beauty", "movielens"]


def run_appendix_alibaba(
    config: RecEngramConfig,
    models: list = None,
    datasets: list = None,
    seeds: list = None,
    output_dir: str = "results/",
):
    """Appendix H: Alibaba baseline results across datasets."""
    if models is None:
        models = ALIBABA_MODELS
    if datasets is None:
        datasets = DATASETS
    if seeds is None:
        seeds = [42, 123, 256]

    output_dir = Path(output_dir) / "appendix" / "alibaba"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for dataset in datasets:
        loader_fn = DATASET_LOADERS[dataset]
        interactions = loader_fn(config)
        config_k = 10 if dataset == "movielens" else 5
        filtered = preprocess_interactions(interactions, k_core=config_k)
        train_data, val_data, test_data = chrono_split(filtered, seed=42)
        vocabs = build_vocabs(train_data)
        item_vocab_size = len(vocabs["item"])

        results[dataset] = {}

        for model_name in models:
            model_seed_results = {}
            print(f"\n{model_name} on {dataset}")

            for seed in seeds:
                torch.manual_seed(seed)
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

                model = build_model(model_name, item_vocab_size)
                result = run_training(
                    model, train_loader, val_loader, test_loader, config,
                    model_name=model_name, verbose=False,
                )

                model_seed_results[f"seed_{seed}"] = {
                    "recall": result["test_metrics"].get("recall", 0),
                    "ndcg": result["test_metrics"].get("ndcg", 0),
                    "hr": result["test_metrics"].get("hr", 0),
                    "mrr": result["test_metrics"].get("mrr", 0),
                }

            recalls = [v["recall"] for v in model_seed_results.values()]
            ndcgs = [v["ndcg"] for v in model_seed_results.values()]
            hrs = [v["hr"] for v in model_seed_results.values()]
            mrrs = [v["mrr"] for v in model_seed_results.values()]
            n = len(recalls)

            results[dataset][model_name] = {
                "recall_mean": sum(recalls) / n,
                "recall_std": (sum((r - sum(recalls)/n)**2 for r in recalls) / n) ** 0.5,
                "ndcg_mean": sum(ndcgs) / n,
                "ndcg_std": (sum((r - sum(ndcgs)/n)**2 for r in ndcgs) / n) ** 0.5,
                "hr_mean": sum(hrs) / n,
                "hr_std": (sum((r - sum(hrs)/n)**2 for r in hrs) / n) ** 0.5,
                "mrr_mean": sum(mrrs) / n,
                "mrr_std": (sum((r - sum(mrrs)/n)**2 for r in mrrs) / n) ** 0.5,
            }

    out_path = output_dir / "alibaba_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def run_appendix_dynamicsasrec(
    config: RecEngramConfig,
    datasets: list = None,
    output_dir: str = "results/",
):
    """Appendix D.3: DynamicSASRec comparison."""
    if datasets is None:
        datasets = DATASETS

    output_dir = Path(output_dir) / "appendix" / "dynamicsasrec"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for dataset in datasets:
        loader_fn = DATASET_LOADERS[dataset]
        interactions = loader_fn(config)
        config_k = 10 if dataset == "movielens" else 5
        filtered = preprocess_interactions(interactions, k_core=config_k)
        train_data, val_data, test_data = chrono_split(filtered, seed=42)
        vocabs = build_vocabs(train_data)
        item_vocab_size = len(vocabs["item"])

        results[dataset] = {}

        for model_name in APPENDIX_MODELS:
            print(f"\n{model_name} on {dataset}")
            torch.manual_seed(42)

            train_ds = SeqRecDataset(train_data, vocabs, config, config.num_negatives)
            val_ds = SeqRecDataset(val_data, vocabs, config, config.num_negatives)
            test_ds = SeqRecDataset(test_data, vocabs, config, config.num_negatives)

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
            )

            if model_name == "recengram":
                from ..models.recengram.model import RecEngram
                from ..training.recengram_trainer import run_training_recengram
                model = RecEngram(item_vocab_size=item_vocab_size, **config.__dict__)
                result = run_training_recengram(
                    model, train_loader, val_loader, test_loader, config, verbose=False,
                )
            else:
                model = build_model(model_name, item_vocab_size)
                result = run_training(
                    model, train_loader, val_loader, test_loader, config,
                    model_name=model_name, verbose=False,
                )

            results[dataset][model_name] = {
                "recall": result["test_metrics"].get("recall", 0),
                "ndcg": result["test_metrics"].get("ndcg", 0),
                "hr": result["test_metrics"].get("hr", 0),
                "mrr": result["test_metrics"].get("mrr", 0),
            }

    out_path = output_dir / "dynamicsasrec_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def run_significance_test(
    config: RecEngramConfig,
    dataset: str = "taobao",
    output_dir: str = "results/",
    n_resamples: int = 10000,
    seed: int = 42,
):
    """Paired bootstrap significance test (paper: 10,000 resamples).

    Compares RecEngram against CL4SRec (strongest published baseline)
    on the Taobao test set. One-sided p-value from bootstrap.
    """
    import numpy as np
    from pathlib import Path
    from ..data.loaders import DATASET_LOADERS
    from ..data.preprocessing import preprocess_interactions, chrono_split, build_vocabs
    from ..data.dataset import SeqRecDataset, collate_fn
    from ..models.recengram.model import RecEngram
    from ..models.cl4srec import CL4SRec
    from ..training.recengram_trainer import run_training_recengram
    from ..training.trainer import run_training
    from ..evaluation.significance import collect_per_user_scores, run_significance_suite

    output_dir = Path(output_dir) / "appendix" / "significance"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_fn = DATASET_LOADERS[dataset]
    interactions = loader_fn(config)
    config_k = 10 if dataset == "movielens" else 5
    filtered = preprocess_interactions(interactions, k_core=config_k)
    train_data, val_data, test_data = chrono_split(filtered, seed=42)
    vocabs = build_vocabs(train_data)
    item_vocab_size = len(vocabs["item"])
    config.item_vocab_size = item_vocab_size

    train_ds = SeqRecDataset(train_data, vocabs, config, config.num_negatives)
    val_ds = SeqRecDataset(val_data, vocabs, config, config.num_negatives)
    test_ds = SeqRecDataset(test_data, vocabs, config, config.num_negatives)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Train RecEngram
    print("Training RecEngram...")
    torch.manual_seed(seed)
    recengram = RecEngram(item_vocab_size=item_vocab_size, **config.__dict__)
    _ = run_training_recengram(
        recengram, train_loader, val_loader, None, config, verbose=False,
    )
    recengram_scores = collect_per_user_scores(recengram, test_loader, device)

    # Train CL4SRec
    print("Training CL4SRec...")
    torch.manual_seed(seed)
    cl4srec = CL4SRec(item_vocab_size=item_vocab_size, hidden_dim=config.hidden_dim,
                      max_seq_len=config.max_seq_len, num_layers=config.num_layers,
                      num_heads=config.num_heads, dropout=config.dropout)
    _ = run_training(
        cl4srec, train_loader, val_loader, None, config,
        model_name="cl4srec", verbose=False,
    )
    cl4srec_scores = collect_per_user_scores(cl4srec, test_loader, device)

    # Run significance tests
    baseline_scores = {"CL4SRec": cl4srec_scores}
    results = run_significance_suite(
        recengram_scores, baseline_scores,
        n_resamples=n_resamples, k=config.top_k, seed=seed,
    )

    # Also compute bootstrap CIs for each model
    from ..evaluation.significance import bootstrap_confidence_interval
    recengram_ci = bootstrap_confidence_interval(
        recengram_scores, n_resamples=n_resamples, k=config.top_k, seed=seed,
    )
    cl4srec_ci = bootstrap_confidence_interval(
        cl4srec_scores, n_resamples=n_resamples, k=config.top_k, seed=seed,
    )

    output = {
        "paired_bootstrap": results,
        "recengram_bootstrap": recengram_ci,
        "cl4srec_bootstrap": cl4srec_ci,
    }

    out_path = output_dir / f"{dataset}_significance.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSignificance test results:")
    for name, r in results.items():
        p = r["p_value"]
        sig = "SIGNIFICANT" if r["significant"] else "NOT significant"
        print(f"  RecEngram vs {name}: p={p:.4f} ({sig})")
        print(f"    RecEngram R@{config.top_k}: {r['model_a_mean']:.4f}")
        print(f"    {name} R@{config.top_k}: {r['model_b_mean']:.4f}")
        print(f"    Diff 95% CI: [{r['ci_95_lower']:.4f}, {r['ci_95_upper']:.4f}]")

    print(f"\nResults saved to {out_path}")
    return output
