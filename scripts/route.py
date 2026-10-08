"""Train intervention routers on validation-period effects and evaluate on the test period.

Policies: no-op, best uniform action, uncertainty-threshold gate, single-stage
selector, gate+selector without harm weighting (h_w=1), harm-aware gate+selector,
and the hindsight oracle. Learned routers are repeated over `routing.seeds`.

Example:
    python scripts/route.py --config configs/experiment/ml-1m_sasrec_reranking.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recint.experiment import run_period_effects
from recint.routing import (
    NO_OP,
    GateSelectorRouter,
    SingleStageRouter,
    best_uniform_choices,
    build_routing_data,
    evaluate_policy,
    oracle_choices,
    uncertainty_gate_choices,
)
from recint.utils import load_experiment_config

DISPLAY_FLOAT_FORMAT = "{:.4f}".format
UNIFORM_POLICY = "best_uniform"
ROUTER_POLICY = "gate_selector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config with a `routing` section")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Dotted config override, e.g. routing.harm_weight=3 (repeatable)",
    )  # fmt: skip
    parser.add_argument("--output-dir", type=Path, help="Overrides `output_dir` in the config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.override)
    routing = config["routing"]
    effects = run_period_effects(config)
    columns = (routing["features"], routing["metric"], routing["tiebreak_metric"])
    valid = build_routing_data(effects.validation, *columns)
    test = build_routing_data(effects.test, *columns)

    rows = [
        {"policy": "no_op", "seed": None, **evaluate_policy(test, np.full(len(test.user_ids), NO_OP))},
        {"policy": UNIFORM_POLICY, "seed": None, **evaluate_policy(test, best_uniform_choices(valid, test))},
        {
            "policy": "uncertainty_gate",
            "seed": None,
            **evaluate_policy(
                test,
                uncertainty_gate_choices(valid, test, routing["uncertainty_feature"], routing["uncertainty_quantiles"]),
            ),
        },
    ]
    thresholds = {}
    for seed in routing["seeds"]:
        router_args = dict(thresholds=routing["thresholds"], classifier=routing["classifier"],
                           n_folds=routing["n_folds"], seed=seed,
                           balance_classes=routing["balance_classes"])  # fmt: skip
        single = SingleStageRouter(routing["classifier"], seed).fit(valid)
        rows.append({"policy": "single_stage", "seed": seed, **evaluate_policy(test, single.choose(test))})
        unweighted = GateSelectorRouter(harm_weight=1.0, **router_args).fit(valid)
        rows.append({"policy": "gate_selector_hw1", "seed": seed, **evaluate_policy(test, unweighted.choose(test))})
        router = GateSelectorRouter(harm_weight=routing["harm_weight"], **router_args).fit(valid)
        rows.append({"policy": ROUTER_POLICY, "seed": seed, **evaluate_policy(test, router.choose(test))})
        thresholds[seed] = router.threshold
    rows.append({"policy": "oracle", "seed": None, **evaluate_policy(test, oracle_choices(test))})

    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("policy", sort=False).agg(
        mean_gain=("mean_gain", "mean"),
        gain_std=("mean_gain", "std"),
        tiebreak_mean_gain=("tiebreak_mean_gain", "mean"),
        n_rescued=("n_rescued", "mean"),
        n_damaged=("n_damaged", "mean"),
        net=("net", "mean"),
        intervene_rate=("intervene_rate", "mean"),
    ).reset_index()
    summary.insert(1, "metric_after", float(test.base.mean()) + summary["mean_gain"])
    uniform_gain = summary.loc[summary["policy"] == UNIFORM_POLICY, "mean_gain"].item()
    # Distance from the best uniform policy in seed standard deviations (learned routers only).
    std = summary["gain_std"].where(summary["gain_std"] > 0)
    summary["sigma_vs_uniform"] = (summary["mean_gain"] - uniform_gain) / std

    run_dir = (args.output_dir or Path(config["output_dir"])) / config["name"] / "routing"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    effects.validation.to_csv(run_dir / "validation_effects.csv", index=False)
    per_seed.to_csv(run_dir / "policies_per_seed.csv", index=False)
    summary.to_csv(run_dir / "policies.csv", index=False)

    with pd.option_context("display.width", None, "display.max_columns", None):
        print(f"Experiment: {config['name']}  routing metric: {routing['metric']} "
              f"(tiebreak {routing['tiebreak_metric']})  base {test.base.mean():.4f}")  # fmt: skip
        print(f"actions: {', '.join(test.actions)}")
        print(f"validation users: {len(valid.user_ids)}  test users: {len(test.user_ids)}  "
              f"gate thresholds per seed: {thresholds}")  # fmt: skip
        print(summary.to_string(index=False, float_format=DISPLAY_FLOAT_FORMAT))
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
