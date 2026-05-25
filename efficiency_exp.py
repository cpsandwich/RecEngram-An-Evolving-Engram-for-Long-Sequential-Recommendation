"""Table 3: Efficiency comparison on Taobao dataset.

Training time (s/epoch), inference latency (ms), GPU memory (MB).
"""

import json
import time
import torch
from pathlib import Path
from ..config import RecEngramConfig
from ..data.loaders import DATASET_LOADERS
from ..data.preprocessing import preprocess_interactions, chrono_split, build_vocabs
from ..data.dataset import SeqRecDataset, collate_fn
from ..models import build_model
from ..evaluation.efficiency import measure_efficiency

EFFICIENCY_MODELS = [
    "gru4rec", "sasrec", "bert4rec", "tisasrec", "hpmn",
    "cl4srec", "din", "dien", "mind", "sdm", "sim", "engram", "recengram",
]


def run_efficiency_experiment(
    config: RecEngramConfig,
    models: list = None,
    dataset: str = "taobao",
    output_dir: str = "results/",
):
    if models is None:
        models = EFFICIENCY_MODELS

    output_dir = Path(output_dir) / "table3"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_fn = DATASET_LOADERS[dataset]
    interactions = loader_fn(config)

    filtered = preprocess_interactions(interactions, k_core=5)
    train_data, val_data, test_data = chrono_split(filtered, seed=42)
    vocabs = build_vocabs(train_data)
    item_vocab_size = len(vocabs["item"])
    config.item_vocab_size = item_vocab_size

    test_ds = SeqRecDataset(test_data, vocabs, config, config.num_negatives, max_seq_len=config.max_seq_len)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers,
    )

    results = {}

    for model_name in models:
        print(f"\nMeasuring efficiency for {model_name}...")
        torch.manual_seed(42)

        model = build_model(model_name, item_vocab_size)
        if model_name in ("pop", "itemknn"):
            continue

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        eff = measure_efficiency(model, test_loader, device)

        train_ds = SeqRecDataset(train_data, vocabs, config, config.num_negatives, max_seq_len=config.max_seq_len)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=config.num_workers,
        )

        model.train()
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            if i >= 50:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _ = model(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time_per_epoch = (time.time() - t0) / 50 * len(train_loader)

        results[model_name] = {
            "training_time_s_per_epoch": train_time_per_epoch,
            "inference_latency_ms": eff["inference_latency_ms"],
            "gpu_memory_mb": eff["gpu_memory_mb"],
        }
        print(f"  Train: {train_time_per_epoch:.1f}s/epoch, "
              f"Infer: {eff['inference_latency_ms']:.2f}ms, "
              f"GPU: {eff['gpu_memory_mb']:.1f}MB")

    out_path = output_dir / "efficiency.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results
