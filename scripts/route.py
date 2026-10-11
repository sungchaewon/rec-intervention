"""Train intervention routers on validation-period effects and evaluate on the test period.

Policies: no-op, best uniform action, uncertainty-threshold gate, single-stage
selector, gate+selector without harm weighting (h_w=1), harm-aware gate+selector,
and the hindsight oracle. Learned routers are repeated over `routing.seeds`.

Example:
    python scripts/route.py --config configs/experiment/ml-1m_sasrec_reranking.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recint.experiment import load_or_compute_period_effects
from recint.routing import (
    NO_OP,
    GainRegressionRouter,
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
    parser.add_argument("--recompute", action="store_true", help="Ignore cached per-user effects")
    return parser.parse_args()


def restrict_actions(user_effects: pd.DataFrame, prefixes: list[str] | None) -> pd.DataFrame:
    """Keep actions whose label prefix (before '@') is listed; all actions if None."""
    if not prefixes:
        return user_effects
    keep = user_effects["action"].str.split("@").str[0].isin(prefixes)
    if not keep.any():
        raise ValueError(f"No actions match routing.action_prefixes={prefixes}")
    return user_effects[keep]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    config = load_experiment_config(args.config, args.override)
    routing = config["routing"]
    experiment_dir = (args.output_dir or Path(config["output_dir"])) / config["name"]
    run_dir = experiment_dir / "routing"
    run_dir.mkdir(parents=True, exist_ok=True)
    effects = load_or_compute_period_effects(config, experiment_dir / "effects", args.recompute)
    logging.info("training routers over seeds %s", routing["seeds"])
    columns = (routing["features"], routing["action_features"], routing["metric"],
               routing["tiebreak_metric"], routing["report_metric"])  # fmt: skip
    prefixes = routing.get("action_prefixes")
    valid = build_routing_data(restrict_actions(effects.validation, prefixes), *columns)
    test = build_routing_data(restrict_actions(effects.test, prefixes), *columns)

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
    for classifier in routing["classifiers"]:
        for seed in routing["seeds"]:
            router_args = dict(thresholds=routing["thresholds"], classifier=classifier,
                               n_folds=routing["n_folds"], seed=seed,
                               balance_classes=routing["balance_classes"])  # fmt: skip
            regression = GainRegressionRouter(classifier, routing["n_folds"], seed).fit(valid)
            rows.append({"policy": f"gain_regression[{classifier}]", "seed": seed,
                         **evaluate_policy(test, regression.choose(test))})  # fmt: skip
            single = SingleStageRouter(classifier, seed).fit(valid)
            unweighted = GateSelectorRouter(harm_weight=1.0, **router_args).fit(valid)
            router = GateSelectorRouter(harm_weight=routing["harm_weight"], **router_args).fit(valid)
            for name, policy in [("single_stage", single), (f"{ROUTER_POLICY}_hw1", unweighted), (ROUTER_POLICY, router)]:
                rows.append({"policy": f"{name}[{classifier}]", "seed": seed, **evaluate_policy(test, policy.choose(test))})
            thresholds[(classifier, seed)] = router.threshold
    rows.append({"policy": "oracle", "seed": None, **evaluate_policy(test, oracle_choices(test))})

    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("policy", sort=False).agg(
        report_gain=("report_gain", "mean"),
        report_gain_std=("report_gain", "std"),
        label_gain=("label_gain", "mean"),
        n_rescued=("n_rescued", "mean"),
        n_damaged=("n_damaged", "mean"),
        net=("net", "mean"),
        intervene_rate=("intervene_rate", "mean"),
    ).reset_index()
    summary.insert(1, "report_after", float(test.report_base.mean()) + summary["report_gain"])
    uniform_gain = summary.loc[summary["policy"] == UNIFORM_POLICY, "report_gain"].item()
    # Distance from the best uniform policy in seed standard deviations (learned routers only).
    std = summary["report_gain_std"].where(summary["report_gain_std"] > 0)
    summary["sigma_vs_uniform"] = (summary["report_gain"] - uniform_gain) / std

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    per_seed.to_csv(run_dir / "policies_per_seed.csv", index=False)
    summary.to_csv(run_dir / "policies.csv", index=False)

    with pd.option_context("display.width", None, "display.max_columns", None):
        print(f"Experiment: {config['name']}  report metric: {routing['report_metric']} "
              f"(base {test.report_base.mean():.4f})  label metric: {routing['metric']}  "
              f"tiebreak: {routing['tiebreak_metric']}")  # fmt: skip
        print("rescued/damaged/net count users on the report metric; label_gain is the label-metric gain")
        print(f"actions: {', '.join(test.actions)}")
        print(f"validation users: {len(valid.user_ids)}  test users: {len(test.user_ids)}  "
              f"gate thresholds per seed: {thresholds}")  # fmt: skip
        print(summary.to_string(index=False, float_format=DISPLAY_FLOAT_FORMAT))
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
