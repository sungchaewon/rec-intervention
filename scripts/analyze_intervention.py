"""Run an intervention-effect analysis from an experiment config.

Prints (1) per-action rescue/damage/net over all users, (2) routing headroom
(best uniform action vs per-user oracle), (3) per-state-group effects, and saves
full tables to the run directory.

Example:
    python scripts/analyze_intervention.py --config configs/experiment/toy_popularity_penalty.yaml
    python scripts/analyze_intervention.py --config ... --override interventions.0.params.penalty_weight=0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from recint.analysis import ALL_USERS
from recint.experiment import run_intervention_experiment
from recint.utils import load_experiment_config

DISPLAY_FLOAT_FORMAT = "{:.4f}".format
ACTION_COLUMNS = ["action", "metric", "n_users", "base", "intervention", "mean_gain", "gain_ci_low",
                  "gain_ci_high", "relative_gain", "n_rescued", "n_damaged", "net", "wilcoxon_p"]  # fmt: skip
GROUP_COLUMNS = ["action", "state", "state_group", "metric", "n_users", "base", "mean_gain",
                 "gain_ci_low", "gain_ci_high", "beneficial_rate", "harm_rate", "net"]  # fmt: skip
HEADROOM_COLUMNS = ["state", "state_group", "metric", "n_users", "best_uniform_action",
                    "best_uniform_gain", "oracle_gain", "oracle_gap", "oracle_intervene_rate"]  # fmt: skip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Dotted config override, e.g. analysis.neutral_threshold=0.01 (repeatable)",
    )  # fmt: skip
    parser.add_argument("--output-dir", type=Path, help="Overrides `output_dir` in the config")
    return parser.parse_args()


def print_section(title: str, table: pd.DataFrame) -> None:
    print(f"\n== {title} ==")
    print(table.to_string(index=False, float_format=DISPLAY_FLOAT_FORMAT))


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.override)
    result = run_intervention_experiment(config)

    run_dir = (args.output_dir or Path(config["output_dir"])) / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    result.user_effects.to_csv(run_dir / "user_effects.csv", index=False)
    result.summary.to_csv(run_dir / "summary.csv", index=False)
    result.headroom.to_csv(run_dir / "headroom.csv", index=False)

    summary = result.summary
    overall = summary["state"] == ALL_USERS
    with pd.option_context("display.width", None, "display.max_columns", None):
        print(f"Experiment: {config['name']}")
        print_section("Per-action effect, all users (for HR@K: rescued/damaged = Rescue@K/Damage@K)",
                      summary.loc[overall, ACTION_COLUMNS])  # fmt: skip
        print_section("Routing headroom (no-op always allowed)", result.headroom[HEADROOM_COLUMNS])
        smallest_k = f"@{min(config['evaluation']['k_values'])}"
        grouped = ~overall & summary["metric"].str.endswith(smallest_k)
        if grouped.any():
            print_section(f"Per-state-group effect ({smallest_k} only; all in summary.csv)",
                          summary.loc[grouped, GROUP_COLUMNS])  # fmt: skip
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
