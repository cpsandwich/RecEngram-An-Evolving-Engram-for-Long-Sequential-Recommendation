"""RecEngram-specific training loop.

Handles:
- Memory write after each batch (StopGrad on memory write per paper Fig 3)
- Surprise-driven evolution after each batch
- MI loss term in addition to BCE loss
- Ablation: surprise/MI only active when use_surprise=True
"""

import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import RecEngramConfig
from ..evaluation.metrics import compute_all_metrics
from ..evaluation.efficiency import measure_efficiency
from .loss import bce_loss_with_negatives


def train_epoch_recengram(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: RecEngramConfig,
) -> float:
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_mi = 0.0
    num_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        all_scores, aux = model(batch)
        h_user = aux["h_user"]

        bce_loss = bce_loss_with_negatives(all_scores, config.l2_reg, model.parameters())

        if model.use_surprise:
            pred_error, surprise_levels = model.compute_surprise(h_user, all_scores)
            mi_loss = model.compute_mi_loss(h_user)
            loss = bce_loss + config.mi_loss_weight * mi_loss
            total_mi += mi_loss.item()
        else:
            loss = bce_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.write_memory(h_user.detach())

        total_loss += loss.item()
        total_bce += bce_loss.item()
        num_batches += 1

    total_loss /= max(num_batches, 1)
    return total_loss


@torch.no_grad()
def evaluate_recengram(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    k: int = 10,
) -> dict:
    model.eval()
    all_recall = []
    all_ndcg = []
    all_hr = []
    all_mrr = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        all_scores, aux = model(batch)
        metrics = compute_all_metrics(all_scores, k=k)
        all_recall.append(metrics[f"Recall@{k}"])
        all_ndcg.append(metrics[f"NDCG@{k}"])
        all_hr.append(metrics[f"HR@{k}"])
        all_mrr.append(metrics["MRR"])

    return {
        "recall": torch.tensor(all_recall).mean().item(),
        "ndcg": torch.tensor(all_ndcg).mean().item(),
        "hr": torch.tensor(all_hr).mean().item(),
        "mrr": torch.tensor(all_mrr).mean().item(),
    }


def run_training_recengram(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: RecEngramConfig,
    verbose: bool = True,
) -> dict:
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
    )

    best_val_ndcg = -1.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_recall": [], "val_ndcg": [], "val_hr": [], "val_mrr": []}

    epoch_iter = range(1, config.num_epochs + 1)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Training RecEngram")

    train_start = time.time()

    for epoch in epoch_iter:
        train_loss = train_epoch_recengram(model, train_loader, optimizer, device, config)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_metrics = evaluate_recengram(model, val_loader, device, config.top_k)
            history["val_recall"].append(val_metrics["recall"])
            history["val_ndcg"].append(val_metrics["ndcg"])
            history["val_hr"].append(val_metrics["hr"])
            history["val_mrr"].append(val_metrics["mrr"])

            if val_metrics["ndcg"] > best_val_ndcg:
                best_val_ndcg = val_metrics["ndcg"]
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.early_stop_patience:
                if verbose:
                    tqdm.write(f"Early stop at epoch {epoch}, best NDCG={best_val_ndcg:.4f}")
                break
        else:
            if best_state is None:
                best_state = copy.deepcopy(model.state_dict())

    train_time = time.time() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = {}
    if test_loader is not None:
        test_metrics = evaluate_recengram(model, test_loader, device, config.top_k)

    eff = measure_efficiency(model, test_loader or val_loader, device)

    if verbose:
        print(f"\nRecEngram results:")
        print(f"  Val NDCG@{config.top_k}: {best_val_ndcg:.4f}")
        if test_metrics:
            for k, v in test_metrics.items():
                print(f"  Test {k}@{config.top_k}: {v:.4f}")
        print(f"  Train time: {train_time:.1f}s")
        print(f"  Inference: {eff['inference_latency_ms']:.2f}ms")
        print(f"  GPU memory: {eff['gpu_memory_mb']:.1f}MB")

    return {
        "best_val_ndcg": best_val_ndcg,
        "test_metrics": test_metrics,
        "train_time_s": train_time,
        "efficiency": eff,
        "history": history,
    }
