"""Table 9: Robustness summary across all scenario analyses.

Aggregates results from Tables 5-8 into a single robustness table.
"""

import json
from pathlib import Path
from ..config import RecEngramConfig
from .scenarios import (
    run_sequence_length_experiment,
    run_concept_drift_experiment,
    run_density_experiment,
    run_cross_category_experiment,
)


def run_robustness_summary(
    config: RecEngramConfig,
    dataset: str = "taobao",
    output_dir: str = "results/",
):
    """Run all scenario experiments and aggregate into robustness summary."""
    output_dir = Path(output_dir) / "table9"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Table 5: Sequence Length ===")
    seq_results = run_sequence_length_experiment(config, dataset, output_dir)

    print("\n=== Table 6: Concept Drift ===")
    drift_results = run_concept_drift_experiment(config, dataset, output_dir)

    print("\n=== Table 7: Interaction Density ===")
    density_results = run_density_experiment(config, dataset, output_dir)

    print("\n=== Table 8: Cross-Category ===")
    cross_cat_results = run_cross_category_experiment(config, dataset, output_dir)

    summary = {
        "sequence_length": seq_results,
        "concept_drift": drift_results,
        "interaction_density": density_results,
        "cross_category": cross_cat_results,
    }

    out_path = output_dir / "robustness_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRobustness summary saved to {out_path}")
    return summary
