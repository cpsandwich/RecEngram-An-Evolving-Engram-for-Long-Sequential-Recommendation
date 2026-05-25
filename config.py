"""Centralized configuration for all models and experiments."""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class RecEngramConfig:
    # ---- Architecture ----
    hidden_dim: int = 128
    max_seq_len: int = 500
    num_memory_slots: int = 128          # total K
    num_long_slots: int = 32             # K_long
    num_short_slots: int = 96            # K_short

    # ---- Embeddings ----
    item_vocab_size: int = 100_000
    category_vocab_size: int = 5_000
    price_vocab_size: int = 10
    brand_vocab_size: int = 50_000

    # ---- Surprise-driven evolution ----
    write_strength: float = 0.1          # eta
    window_size: int = 20                # w
    surprise_low_percentile: float = 0.30
    surprise_high_percentile: float = 0.70
    surprise_window: int = 5000
    long_update_interval: int = 5        # Delta t_long

    # ---- MI critic ----
    critic_hidden_dim: int = 256
    mi_loss_weight: float = 0.1

    # ---- Training ----
    batch_size: int = 256
    learning_rate: float = 1e-3
    num_epochs: int = 100
    num_negatives: int = 8
    l2_reg: float = 1e-5
    temperature: float = 1.0

    # ---- Optimizer ----
    optimizer: str = "adam"
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    early_stop_patience: int = 20

    # ---- Transformer config (shared by SASRec, BERT4Rec, etc.) ----
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.2

    # ---- Hardware ----
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 4

    # ---- Evaluation ----
    top_k: int = 10
    num_seeds: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123, 256, 512, 1024])

    # ---- Dataset paths ----
    taobao_path: str = "data/taobao/"
    amazon_beauty_path: str = "data/amazon_beauty/"
    movielens_path: str = "data/movielens1m/"

    # ---- Output ----
    checkpoint_dir: str = "checkpoints/"
    results_dir: str = "results/"
