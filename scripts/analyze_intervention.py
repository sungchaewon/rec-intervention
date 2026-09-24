"""Run a state-conditioned intervention-effect analysis from an experiment config.

Example:
    python scripts/analyze_intervention.py --config configs/experiment/toy_popularity_penalty.yaml
    python scripts/analyze_intervention.py --config ... --override analysis.neutral_threshold=0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from recint.experiment import run_intervention_experiment
from recint.utils import load_experiment_config

DISPLAY_FLOAT_FORMAT = "{:.4f}".format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. intervention.params.penalty_weight=1.0 (repeatable)",
    )
    parser.add_argument("--output-dir", type=Path, help="Overrides `output_dir` in the config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.override)
    result = run_intervention_experiment(config)

    run_dir = (args.output_dir or Path(config["output_dir"])) / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    result.user_effects.to_csv(run_dir / "user_effects.csv", index=False)
    result.state_summary.to_csv(run_dir / "state_summary.csv", index=False)

    with pd.option_context("display.width", None, "display.max_columns", None):
        print(f"Experiment: {config['name']}  intervention: {config['intervention']['name']}")
        print(result.state_summary.to_string(index=False, float_format=DISPLAY_FLOAT_FORMAT))
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
