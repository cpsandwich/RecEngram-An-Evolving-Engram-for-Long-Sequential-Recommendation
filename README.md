# RecEngram: Dynamic Memory for Sequential Recommendation

Official implementation of RecEngram, a dynamic memory-augmented framework for sequential recommendation. The model combines three mutually-reinforcing mechanisms: **Target-Guided Memory Activation**, **Surprise-Driven Memory Evolution**, and **Hierarchical Long-Short Term Memory**.

This repository includes 16 models (10 main baselines + 5 Alibaba industrial baselines + DynamicSASRec) and experiment scripts for all 9 main tables and 7 appendix tables.

## Installation

```bash
git clone [https://github.com/cpsandwich/RecEngram-An-Evolving-Engram-for-Long-Sequential-Recommendation.git]
cd recengram
pip install -e .
```

Requirements: `torch >= 2.0`, `numpy`, `tqdm`.

## Datasets

Place datasets under the `data/` directory:

| Dataset | Directory | Required Files |
|---------|-----------|---------------|
| Taobao | `data/taobao/` | `UserBehavior.csv` (uid, iid, cid, brand_id, timestamp) |
| Amazon Beauty | `data/amazon_beauty/` | `reviews_Beauty_5.json.gz` ([download](http://jmcauley.ucsd.edu/data/amazon/)) |
| MovieLens-1M | `data/movielens1m/` | `ratings.dat` ([download](https://grouplens.org/datasets/movielens/1m/))

## Usage

Run experiments via the unified CLI:

```bash
# Table 2: Overall performance (10 models x 3 datasets x 5 seeds)
python -m recengram.cli table2 --dataset taobao --models pop,sasrec

# Table 3: Efficiency comparison
python -m recengram.cli table3

# Table 4: Ablation study
python -m recengram.cli ablation --dataset taobao --experiment additive
python -m recengram.cli ablation --dataset amazon_beauty --experiment subtractive
python -m recengram.cli ablation --experiment slots

# Tables 5-8: Scenario analyses
python -m recengram.cli scenarios --dataset taobao --scenario seq_length
python -m recengram.cli scenarios --scenario drift
python -m recengram.cli scenarios --scenario density
python -m recengram.cli scenarios --scenario cross_category

# Table 9: Robustness summary
python -m recengram.cli robustness

# Appendix experiments
python -m recengram.cli appendix --section alibaba
python -m recengram.cli appendix --section dynamicsasrec

# Run all experiments
python -m recengram.cli all
```

## Models

| Category | Model | Description |
|----------|-------|-------------|
| Traditional | POP | Global item popularity |
| Traditional | ItemKNN | Sparse item-item co-occurrence |
| Deep Learning | GRU4Rec | Multi-layer GRU over sequence |
| Deep Learning | SASRec | Causal Transformer decoder |
| Deep Learning | BERT4Rec | Bidirectional Transformer |
| Time-Aware | TiSASRec | Time-interval attention with gamma decay |
| Time-Aware | DynamicSASRec | SASRec + learnable temporal decay per head |
| Complex | HPMN | Hierarchical periodic memory with multi-hop read |
| Contrastive | CL4SRec | SASRec + 3 augmentations + InfoNCE |
| Alibaba | DIN | Adaptive attention with Dice activation |
| Alibaba | DIEN | Interest extractor + AUGRU evolving |
| Alibaba | MIND | Multi-interest capsule routing |
| Alibaba | SDM | Session attention + user encoder |
| Alibaba | SIM | Two-stage GSU hard search + ESU exact attention |
| Ours | Engram | Base memory-augmented model |
| Ours | RecEngram | Full model with all 3 mechanisms |

## Configuration

All hyperparameters are centralized in `recengram/config.py` (`RecEngramConfig` dataclass):

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 128 | Hidden dimension |
| num_memory_slots | 128 | Total memory slots |
| num_long_slots | 32 | Long-term bank slots |
| num_short_slots | 96 | Short-term bank slots |
| num_layers | 2 | Transformer layers |
| num_heads | 2 | Attention heads |
| dropout | 0.2 | Dropout rate |
| batch_size | 256 | Training batch size |
| learning_rate | 1e-3 | Adam learning rate |
| num_epochs | 100 | Max training epochs |
| num_negatives | 8 | Negative samples per positive |
| early_stop_patience | 20 | Early stopping patience |
| seeds | [42, 123, 256, 512, 1024] | Random seeds for 5-run average |

## Project Structure

```
recengram_v2/
├── recengram/
│   ├── config.py              # Global configuration
│   ├── cli.py                 # CLI entry point
│   ├── data/                  # Data pipeline
│   │   ├── loaders.py         # Dataset loaders (Taobao, Amazon, MovieLens)
│   │   ├── preprocessing.py   # k-core filtering, chrono split, vocab
│   │   └── dataset.py         # PyTorch Dataset + collate
│   ├── models/                # 16 models
│   │   ├── common.py          # Shared building blocks
│   │   ├── pop.py, itemknn.py # Traditional baselines
│   │   ├── gru4rec.py, sasrec.py, bert4rec.py  # Deep learning
│   │   ├── tisasrec.py, dynamicsasrec.py        # Time-aware
│   │   ├── hpmn.py, cl4srec.py                  # Complex
│   │   ├── din.py, dien.py, mind.py, sdm.py, sim.py  # Alibaba
│   │   └── recengram/         # RecEngram core
│   │       ├── embedding.py   # Input embeddings
│   │       ├── engram.py      # Memory with content-based R/W
│   │       ├── target_guided.py  # Bilinear target-guided activation
│   │       ├── surprise.py    # Surprise evolution + MI critic
│   │       ├── hierarchical.py   # Dual-bank hierarchical memory
│   │       └── model.py       # Full RecEngram model
│   ├── training/              # Training loops
│   │   ├── loss.py            # BCE, BPR, InfoNCE losses
│   │   ├── trainer.py         # Generic trainer
│   │   └── recengram_trainer.py  # RecEngram-specific trainer
│   ├── evaluation/            # Metrics and efficiency
│   │   ├── metrics.py         # Recall@K, NDCG@K, MRR
│   │   └── efficiency.py      # Latency, GPU memory
│   └── experiments/           # Experiment runners
│       ├── overall.py         # Table 2
│       ├── efficiency_exp.py  # Table 3
│       ├── ablation.py        # Table 4
│       ├── scenarios.py       # Tables 5-8
│       ├── robustness.py      # Table 9
│       └── appendix.py        # Appendix experiments
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```
