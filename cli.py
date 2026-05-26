"""Unified CLI: recengram <experiment> [options].

Usage:
    python -m recengram.cli table2 --dataset taobao --models pop,sasrec
    python -m recengram.cli table3
    python -m recengram.cli ablation --experiment additive
    python -m recengram.cli scenarios --dataset amazon_beauty
    python -m recengram.cli robustness
    python -m recengram.cli appendix --section alibaba
    python -m recengram.cli all
"""

import argparse
import sys
from .config import RecEngramConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="RecEngram: Dynamic Memory for Sequential Recommendation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Experiment to run")

    # Table 2: Overall performance
    p2 = subparsers.add_parser("table2", help="Overall performance (Table 2)")
    p2.add_argument("--dataset", default="taobao", choices=["taobao", "amazon_beauty", "movielens"])
    p2.add_argument("--models", default=None, help="Comma-separated model names")
    p2.add_argument("--seeds", default=None, help="Comma-separated seeds")
    p2.add_argument("--epochs", type=int, default=100)
    p2.add_argument("--output", default="results/")
    p2.add_argument("--checkpoint", default=None, help="Path to pre-trained checkpoint")

    # Table 3: Efficiency
    p3 = subparsers.add_parser("table3", help="Efficiency comparison (Table 3)")
    p3.add_argument("--dataset", default="taobao", choices=["taobao", "amazon_beauty", "movielens"])
    p3.add_argument("--models", default=None, help="Comma-separated model names")
    p3.add_argument("--output", default="results/")

    # Table 4: Ablation
    p4 = subparsers.add_parser("ablation", help="Ablation study (Table 4)")
    p4.add_argument("--dataset", default="taobao", choices=["taobao", "amazon_beauty", "movielens"])
    p4.add_argument("--experiment", default="additive",
                    choices=["additive", "subtractive", "slots"])
    p4.add_argument("--output", default="results/")

    # Tables 5-8: Scenarios
    p5 = subparsers.add_parser("scenarios", help="Scenario analyses (Tables 5-8)")
    p5.add_argument("--dataset", default="taobao", choices=["taobao", "amazon_beauty", "movielens"])
    p5.add_argument("--scenario", default="all",
                    choices=["all", "seq_length", "drift", "density", "cross_category"])
    p5.add_argument("--output", default="results/")

    # Table 9: Robustness
    p9 = subparsers.add_parser("robustness", help="Robustness summary (Table 9)")
    p9.add_argument("--dataset", default="taobao", choices=["taobao", "amazon_beauty", "movielens"])
    p9.add_argument("--output", default="results/")

    # Appendix
    pa = subparsers.add_parser("appendix", help="Appendix experiments")
    pa.add_argument("--section", default="all",
                    choices=["all", "alibaba", "dynamicsasrec", "std", "significance"])
    pa.add_argument("--dataset", default=None)
    pa.add_argument("--output", default="results/")

    # Run all experiments
    pall = subparsers.add_parser("all", help="Run all experiments (Tables 2-9 + Appendix)")

    return parser.parse_args()


def main():
    args = parse_args()
    config = RecEngramConfig()

    if args.command == "table2":
        from .experiments.overall import run_overall_experiment
        models = args.models.split(",") if args.models else None
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        config.num_epochs = args.epochs
        run_overall_experiment(config, args.dataset, models, seeds, args.output, args.checkpoint)

    elif args.command == "table3":
        from .experiments.efficiency_exp import run_efficiency_experiment
        models = args.models.split(",") if args.models else None
        run_efficiency_experiment(config, models, args.dataset, args.output)

    elif args.command == "ablation":
        from .experiments.ablation import run_ablation
        run_ablation(config, args.dataset, args.experiment, args.output)

    elif args.command == "scenarios":
        from .experiments.scenarios import (
            run_sequence_length_experiment,
            run_concept_drift_experiment,
            run_density_experiment,
            run_cross_category_experiment,
        )
        scenario = args.scenario
        if scenario in ("all", "seq_length"):
            run_sequence_length_experiment(config, args.dataset, args.output)
        if scenario in ("all", "drift"):
            run_concept_drift_experiment(config, args.dataset, args.output)
        if scenario in ("all", "density"):
            run_density_experiment(config, args.dataset, args.output)
        if scenario in ("all", "cross_category"):
            run_cross_category_experiment(config, args.dataset, args.output)

    elif args.command == "robustness":
        from .experiments.robustness import run_robustness_summary
        run_robustness_summary(config, args.dataset, args.output)

    elif args.command == "appendix":
        from .experiments.appendix import (
            run_appendix_alibaba, run_appendix_dynamicsasrec,
            run_significance_test,
        )
        section = args.section
        if section in ("all", "alibaba"):
            datasets = [args.dataset] if args.dataset else None
            run_appendix_alibaba(config, datasets=datasets, output_dir=args.output)
        if section in ("all", "dynamicsasrec"):
            datasets = [args.dataset] if args.dataset else None
            run_appendix_dynamicsasrec(config, datasets=datasets, output_dir=args.output)
        if section in ("all", "significance"):
            run_significance_test(config, dataset=args.dataset or "taobao", output_dir=args.output)

    elif args.command == "all":
        print("Running all experiments (Tables 2-9 + Appendix)...")
        from .experiments.overall import run_overall_experiment
        from .experiments.efficiency_exp import run_efficiency_experiment
        from .experiments.ablation import run_ablation
        from .experiments.scenarios import (
            run_sequence_length_experiment,
            run_concept_drift_experiment,
            run_density_experiment,
            run_cross_category_experiment,
        )
        from .experiments.robustness import run_robustness_summary
        from .experiments.appendix import run_appendix_alibaba, run_appendix_dynamicsasrec

        for dataset in ["taobao", "amazon_beauty", "movielens"]:
            print(f"\n{'#'*60}\n# Dataset: {dataset}\n{'#'*60}")
            config.num_epochs = 100
            run_overall_experiment(config, dataset, output_dir="results/")
            run_efficiency_experiment(config, dataset=dataset, output_dir="results/")
            run_ablation(config, dataset, "additive", "results/")
            run_ablation(config, dataset, "subtractive", "results/")
            run_ablation(config, dataset, "slots", "results/")
            run_sequence_length_experiment(config, dataset, "results/")
            run_concept_drift_experiment(config, dataset, "results/")
            run_density_experiment(config, dataset, "results/")
            run_cross_category_experiment(config, dataset, "results/")
            run_robustness_summary(config, dataset, "results/")
            run_appendix_alibaba(config, datasets=[dataset], output_dir="results/")
            run_appendix_dynamicsasrec(config, datasets=[dataset], output_dir="results/")

        print("\nAll experiments complete!")

    else:
        print("No command specified. Use --help for available commands.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
