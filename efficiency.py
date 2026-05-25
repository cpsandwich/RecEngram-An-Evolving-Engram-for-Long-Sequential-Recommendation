"""Efficiency measurement: training time, inference latency, GPU memory."""

import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def measure_efficiency(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup: int = 100,
    num_measure: int = 1000,
) -> Dict[str, float]:
    """Measure per-step inference latency and peak GPU memory.

    Matches paper efficiency protocol (Appendix G): warmup 100 steps,
    measure 1000 steps with cuda.synchronize().

    Returns: {"inference_ms": float, "peak_memory_mb": float}
    """
    model.eval()
    is_cuda = device.type == "cuda"

    # Reset memory stats
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= num_warmup:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.no_grad():
            _ = model(batch)

    if is_cuda:
        torch.cuda.synchronize()

    # Measure
    t0 = time.time()
    for i, batch in enumerate(dataloader):
        if i >= num_measure:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.no_grad():
            _ = model(batch)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    inference_ms = (elapsed / num_measure) * 1000

    if is_cuda:
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_mem = 0.0

    return {"inference_ms": inference_ms, "peak_memory_mb": peak_mem}


def measure_training_time(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 5,
    skip_first: bool = True,
    train_step_fn=None,
) -> float:
    """Measure average per-epoch training time (paper Appendix G).

    Runs num_epochs, skips first epoch's time (initialization overhead),
    returns mean seconds/epoch for remaining epochs.
    """
    model.train()
    is_cuda = device.type == "cuda"
    epoch_times = []

    for epoch in range(num_epochs):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            if train_step_fn:
                train_step_fn(model, batch, optimizer, device)
            else:
                optimizer.zero_grad()
                output = model(batch)
                if isinstance(output, tuple):
                    output = output[0]
                loss = output.mean()
                loss.backward()
                optimizer.step()

        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        if is_cuda and epoch == 0:
            torch.cuda.reset_peak_memory_stats(device)

    start = 1 if skip_first and len(epoch_times) > 1 else 0
    return sum(epoch_times[start:]) / len(epoch_times[start:])
