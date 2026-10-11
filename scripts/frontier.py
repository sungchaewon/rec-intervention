"""Harm-aware routing of beyond-accuracy re-rankers: matched comparison with uniform strengths.

For each re-ranker family and each uniform strength, routers are tuned on the
validation period to reach that strength's objective gain (e.g. long-tail
share), then compared on the test period: same objective, how much accuracy
is lost and how many users are damaged? Also writes full frontier curves.

Example:
    python scripts/frontier.py --config configs/experiment/ml-1m_sasrec_reranking.yaml
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recint.analysis import bootstrap_mean_ci
from recint.experiment import action_label, expand_interventions, load_or_compute_period_effects
from recint.routing.policy import realized
from recint.routing.frontier import (
    DamageModel,
    FamilyData,
    build_family_data,
    evaluate_choices,
    gate_choices,
    match_threshold,
    strongest_safe,
    uniform_choices,
)
from recint.utils import load_experiment_config

DISPLAY_FLOAT_FORMAT = "{:.4f}".format
UNIFORM = "uniform"
SUMMARY_COLUMNS = ["family", "target", "policy", "objective_gain", "report_gain", "n_damaged", "n_rescued",
                   "intervene_rate", "report_gain_vs_uniform", "gain_diff_ci_low", "gain_diff_ci_high",
                   "damage_diff", "damage_diff_ci_low", "damage_diff_ci_high", "damage_reduction"]  # fmt: skip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config with a `frontier` section")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Dotted config override, e.g. frontier.harm_weights=[1,3] (repeatable)",
    )  # fmt: skip
    parser.add_argument("--output-dir", type=Path, help="Overrides `output_dir` in the config")
    parser.add_argument("--recompute", action="store_true", help="Ignore cached per-user effects")
    return parser.parse_args()


def user_feature(data: FamilyData, user_features: list[str], name: str) -> np.ndarray:
    return data.features[:, 0, user_features.index(name)]


def quantile_thresholds(p_damage: np.ndarray, n: int) -> np.ndarray:
    """Candidate taus at quantiles of the predicted damage probabilities (fine control of
    how many (user, strength) pairs are allowed), plus one below all for the no-op."""
    return np.r_[-1.0, np.unique(np.quantile(p_damage, np.linspace(0.0, 1.0, n)))]


def run_family(
    family: dict, actions: list[str], effects, settings: dict
) -> tuple[list[dict], list[dict], dict[tuple, list[np.ndarray]]]:
    """Returns matched rows, test frontier curves, and per-user test report gains per
    (family, target, policy), one array per seed, for paired comparisons."""
    columns = (settings["features"], settings["action_features"], family["objective"],
               settings["label_metric"], settings["report_metric"])  # fmt: skip
    valid = build_family_data(effects.validation, actions, *columns)
    test = build_family_data(effects.test, actions, *columns)
    matched, curves = [], []
    per_user: dict[tuple, list[np.ndarray]] = {}

    # Uncertainty-gate heuristic: (strength, quantile) chosen on validation per target.
    feature = settings["uncertainty_feature"]
    valid_values = user_feature(valid, settings["features"], feature)
    test_values = user_feature(test, settings["features"], feature)
    gate_grid = []
    for action, quantile in itertools.product(range(len(actions)), settings["uncertainty_quantiles"]):
        cut = float(np.quantile(valid_values, quantile))
        gate_grid.append((action, cut, evaluate_choices(valid, gate_choices(valid_values, cut, action))))
        curves.append({"policy": "uncertainty_gate", "seed": None, "setting": f"{actions[action]}|q={quantile}",
                       **evaluate_choices(test, gate_choices(test_values, cut, action))})  # fmt: skip

    routers = {}
    for classifier, harm_weight, seed in itertools.product(
        settings["classifiers"], settings["harm_weights"], settings["seeds"]
    ):
        model = DamageModel(classifier, harm_weight, seed).fit(valid)
        p_valid, p_test = model.predict(valid), model.predict(test)
        name = f"router[{classifier},hw={harm_weight:g}]"
        thresholds = quantile_thresholds(p_valid, settings["n_thresholds"])
        valid_objective = {t: evaluate_choices(valid, strongest_safe(p_valid, t))["objective_gain"] for t in thresholds}
        routers[(name, seed)] = (p_test, valid_objective)
        for t in thresholds:
            curves.append({"policy": name, "seed": seed, "setting": f"tau={t:.3f}",
                           **evaluate_choices(test, strongest_safe(p_test, t))})  # fmt: skip

    for target, action in enumerate(actions):
        target_objective = evaluate_choices(valid, uniform_choices(valid, target))["objective_gain"]
        base_row = {"family": family["prefix"], "target": action}
        uniform = evaluate_choices(test, uniform_choices(test, target))
        matched.append({**base_row, "policy": UNIFORM, "seed": None, **uniform})
        key = (family["prefix"], action)
        per_user[(*key, UNIFORM)] = [realized(test.report_gain, uniform_choices(test, target))]
        curves.append({"policy": UNIFORM, "seed": None, "setting": action, **uniform})

        reaching = [g for g in gate_grid if g[2]["objective_gain"] >= target_objective] or gate_grid
        gate_action, gate_cut, _ = max(reaching, key=lambda g: g[2]["report_gain"])
        gate = gate_choices(test_values, gate_cut, gate_action)
        matched.append({**base_row, "policy": "uncertainty_gate", "seed": None, **evaluate_choices(test, gate)})
        per_user[(*key, "uncertainty_gate")] = [realized(test.report_gain, gate)]

        for (name, seed), (p_test, valid_objective) in routers.items():
            choices = strongest_safe(p_test, match_threshold(valid_objective, target_objective))
            matched.append({**base_row, "policy": name, "seed": seed, **evaluate_choices(test, choices)})
            per_user.setdefault((*key, name), []).append(realized(test.report_gain, choices))
    for row in curves:
        row["family"] = family["prefix"]
    return matched, curves, per_user


def paired_comparison(
    per_user: dict[tuple, list[np.ndarray]], n_resamples: int, confidence: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Per user, policy (seed-averaged) minus uniform at the same target: bootstrap CIs of the
    mean report-gain difference and of the damaged-user count difference (negative = fewer)."""
    rows = []
    for (family, target, policy), gains in per_user.items():
        if policy == UNIFORM:
            continue
        uniform = per_user[(family, target, UNIFORM)][0]
        gain = np.mean(gains, axis=0)
        damage = np.mean([g < 0 for g in gains], axis=0)
        gain_diff, damage_diff = gain - uniform, damage - (uniform < 0)
        n_users = len(uniform)
        gain_low, gain_high = bootstrap_mean_ci(gain_diff, n_resamples, confidence, rng)
        damage_low, damage_high = bootstrap_mean_ci(damage_diff, n_resamples, confidence, rng)
        rows.append({
            "family": family, "target": target, "policy": policy,
            "gain_diff_ci_low": gain_low,
            "gain_diff_ci_high": gain_high,
            "damage_diff": damage_diff.sum(),
            "damage_diff_ci_low": damage_low * n_users,
            "damage_diff_ci_high": damage_high * n_users,
        })  # fmt: skip
    return pd.DataFrame(rows)


def summarize(matched: pd.DataFrame) -> pd.DataFrame:
    summary = matched.groupby(["family", "target", "policy"], sort=False).agg(
        objective_gain=("objective_gain", "mean"),
        report_gain=("report_gain", "mean"),
        report_gain_std=("report_gain", "std"),
        n_damaged=("n_damaged", "mean"),
        n_rescued=("n_rescued", "mean"),
        intervene_rate=("intervene_rate", "mean"),
    ).reset_index()
    uniform = summary[summary["policy"] == UNIFORM].set_index(["family", "target"])
    key = list(zip(summary["family"], summary["target"]))
    summary["report_gain_vs_uniform"] = summary["report_gain"].to_numpy() - uniform.loc[key, "report_gain"].to_numpy()
    uniform_damage = uniform.loc[key, "n_damaged"].to_numpy()
    summary["damage_reduction"] = np.divide(
        uniform_damage - summary["n_damaged"].to_numpy(), uniform_damage,
        out=np.full(len(summary), np.nan), where=uniform_damage > 0,
    )  # fmt: skip
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    config = load_experiment_config(args.config, args.override)
    settings = config["frontier"]
    experiment_dir = (args.output_dir or Path(config["output_dir"])) / config["name"]
    effects = load_or_compute_period_effects(config, experiment_dir / "effects", args.recompute)
    labels = [action_label(c) for c in expand_interventions(config["interventions"])]

    matched, curves, per_user = [], [], {}
    for family in settings["families"]:
        actions = [label for label in labels if label.split("@")[0] == family["prefix"]]
        if not actions:
            raise ValueError(f"No actions labeled {family['prefix']}@... in `interventions`")
        logging.info("family %s: %s (objective %s)", family["prefix"], actions, family["objective"])
        family_matched, family_curves, family_per_user = run_family(family, actions, effects, settings)
        matched += family_matched
        curves += family_curves
        per_user.update(family_per_user)

    run_dir = experiment_dir / "frontier"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    matched = pd.DataFrame(matched)
    bootstrap = config["analysis"]["bootstrap"]
    paired = paired_comparison(
        per_user, bootstrap["n_resamples"], bootstrap["confidence"], np.random.default_rng(config["seed"])
    )
    summary = summarize(matched).merge(paired, on=["family", "target", "policy"], how="left")
    matched.to_csv(run_dir / "matched_per_seed.csv", index=False)
    summary.to_csv(run_dir / "matched.csv", index=False)
    pd.DataFrame(curves).to_csv(run_dir / "curves.csv", index=False)

    with pd.option_context("display.width", None, "display.max_columns", None, "display.max_rows", None):
        print(f"Experiment: {config['name']}  report metric: {settings['report_metric']}  "
              f"damage label: {settings['label_metric']} < 0")  # fmt: skip
        print("Routers are tuned on validation to match each uniform strength's objective gain; test results below.")
        print("Paired vs uniform at the same target (95% bootstrap CI over users; routers averaged over seeds):")
        print("  report_gain_vs_uniform > 0: less accuracy lost; damage_diff < 0: fewer damaged users.")
        print(summary[SUMMARY_COLUMNS].to_string(index=False, float_format=DISPLAY_FLOAT_FORMAT))
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
