"""End-to-end observed-intervention-effect experiment.

data -> split -> base model -> user states (from base scores)
     -> each intervention action -> per-user metrics -> gain
     -> per-action / per-state summary + routing headroom
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recint.analysis import build_user_effect_table, routing_headroom, summarize_effects
from recint.data import Dataset, EvaluationSplit, InteractionData, leave_last_out, load_dataset
from recint.interventions import build_intervention
from recint.metrics import compute_ranking_metrics, metric_name
from recint.metrics.beyond_accuracy import ListQualityMetrics
from recint.models import Recommender, build_model, load_model, mask_seen_items, top_k_items
from recint.states import StateContext, assign_state_groups, build_state_extractor
from recint.utils import make_rngs, set_global_seed

logger = logging.getLogger(__name__)

RNG_STREAMS = ["data", "analysis"]


@dataclass(frozen=True)
class ExperimentResult:
    user_effects: pd.DataFrame  # one row per (user, action)
    summary: pd.DataFrame  # per (action, state, group, metric)
    headroom: pd.DataFrame  # per (state, group, metric): best uniform vs oracle


def prepare_run(
    config: dict[str, Any],
) -> tuple[Dataset, EvaluationSplit, dict[str, np.random.Generator]]:
    """Seed everything, load data, and split. Shared by training and analysis so
    both always see the same train/test split."""
    seed = config["seed"]
    set_global_seed(seed)
    rngs = make_rngs(seed, RNG_STREAMS)
    dataset = load_dataset(config["dataset"], rngs["data"])
    split = leave_last_out(dataset.interactions, config["dataset"]["split"]["min_train_interactions"])
    if len(split.eval_users) == 0:
        raise ValueError("No users have enough interactions for evaluation")
    return dataset, split, rngs


def obtain_base_model(config: dict[str, Any], train: InteractionData) -> Recommender:
    """Load `model_checkpoint` if configured (trained by scripts/train.py), else fit."""
    checkpoint = config.get("model_checkpoint")
    if checkpoint is None:
        return build_model(config["model"]).fit(train)
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; train it first with scripts/train.py")
    return load_model(config["model"], path, train)


def action_label(intervention_config: dict[str, Any]) -> str:
    return intervention_config.get("label", intervention_config["name"])


def expand_interventions(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand `grid: {param: [values]}` entries into one action per combination,
    labeled e.g. `calibration@0.25` (several grid params: `mmr@0.8,50`)."""
    expanded = []
    for config in configs:
        grid = config.get("grid")
        if not grid:
            expanded.append(config)
            continue
        for values in itertools.product(*grid.values()):
            expanded.append(
                {
                    "name": config["name"],
                    "label": f"{action_label(config)}@{','.join(str(v) for v in values)}",
                    "params": {**config.get("params", {}), **dict(zip(grid, values))},
                }
            )
    return expanded


def validate_interventions(interventions: list[dict[str, Any]], k_values: list[int]) -> None:
    labels = [action_label(c) for c in interventions]
    if not labels:
        raise ValueError("`interventions` must list at least one action")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Intervention labels must be unique (set `label`), got {labels}")
    for intervention_config in interventions:
        # Rerankers can only return their candidates; top-K beyond that is ill-defined.
        candidate_size = intervention_config.get("params", {}).get("candidate_size")
        if candidate_size is not None and candidate_size < max(k_values):
            raise ValueError(
                f"{action_label(intervention_config)}: candidate_size={candidate_size} "
                f"must be >= max k {max(k_values)}"
            )


def compute_states(
    state_configs: list[dict[str, Any]], context: StateContext
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    """State values for every configured state, keyed by `label` (default: `name`);
    groups only for states with `grouping`."""
    labels = [c.get("label", c["name"]) for c in state_configs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"State labels must be unique (set `label`), got {labels}")
    values, groups, groupings = {}, {}, {}
    for name, state_config in zip(labels, state_configs):
        values[name] = build_state_extractor(state_config).compute(context)
        grouping = state_config.get("grouping")
        if grouping is not None:
            groups[name] = assign_state_groups(
                values[name], grouping["method"], grouping["cut_points"], grouping["labels"]
            )
            groupings[name] = list(grouping["labels"])
    return values, groups, groupings


def list_change_features(base_ranking: np.ndarray, intervened_top: np.ndarray) -> dict[str, np.ndarray]:
    """How an action changes each user's top-K (K = intervened_top width), from the
    lists alone (no targets): `changed@K` = items newly entering the top-K, and
    `promoted_rank@K` = their mean 1-indexed base rank (0 if none)."""
    k = intervened_top.shape[1]
    base_top = base_ranking[:, :k]
    entering = ~(intervened_top[:, :, None] == base_top[:, None, :]).any(axis=2)
    match = intervened_top[:, :, None] == base_ranking[:, None, :]
    base_rank = np.where(match.any(axis=2), match.argmax(axis=2) + 1, base_ranking.shape[1] + 1)
    changed = entering.sum(axis=1)
    promoted_rank = np.divide(
        np.where(entering, base_rank, 0).sum(axis=1), changed,
        out=np.zeros(len(changed)), where=changed > 0,
    )  # fmt: skip
    return {f"changed@{k}": changed.astype(np.float64), f"promoted_rank@{k}": promoted_rank}


def evaluate_period(
    config: dict[str, Any],
    interventions: list[dict[str, Any]],
    dataset: Dataset,
    model: Recommender,
    period: EvaluationSplit,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Per-user effects of every action for one evaluation period.

    `period.train` is the history the base model, the interventions, and the
    states see; `period.targets` are the held-out items.
    """
    users, history = period.eval_users, period.train
    # Dense (n_eval_users, n_items) scores: fine up to ML-1M scale; batch for larger catalogs.
    base_scores = model.score(users, history=history)
    mask_seen_items(base_scores, history.user_item_matrix(), users)
    state_values, state_groups, groupings = compute_states(
        config["states"], StateContext(history, users, base_scores, dataset.item_categories)
    )

    evaluation = config["evaluation"]
    k_values, metrics = evaluation["k_values"], evaluation["metrics"]
    max_k, feature_k = max(k_values), min(k_values)
    # Re-rankers only reorder base candidates, so this depth locates every promoted item.
    depth = max([max_k] + [c.get("params", {}).get("candidate_size", max_k) for c in interventions])
    base_ranking = top_k_items(base_scores, depth)
    base_metrics = compute_ranking_metrics(base_ranking[:, :max_k], period.targets, metrics, k_values)
    beyond = evaluation.get("beyond_accuracy")
    quality = None
    if beyond:
        quality = ListQualityMetrics(history, dataset.item_categories, beyond["metrics"], beyond["head_fraction"])
        base_metrics |= quality.compute(users, base_ranking, k_values)

    tables = []
    for position, intervention_config in enumerate(interventions, start=1):
        started = time.perf_counter()
        intervention = build_intervention(intervention_config).fit(history, dataset.item_categories)
        intervened_ranking = top_k_items(intervention.apply(users, base_scores), max_k)
        intervention_metrics = compute_ranking_metrics(intervened_ranking, period.targets, metrics, k_values)
        if quality is not None:
            intervention_metrics |= quality.compute(users, intervened_ranking, k_values)
        tables.append(
            build_user_effect_table(
                user_ids=dataset.interactions.user_ids[users],
                state_values=state_values,
                state_groups=state_groups,
                base_metrics=base_metrics,
                intervention_metrics=intervention_metrics,
                neutral_threshold=config["analysis"]["neutral_threshold"],
                action=action_label(intervention_config),
                action_features=list_change_features(base_ranking, intervened_ranking[:, :feature_k]),
            )
        )
        logger.info(
            "%s (%d/%d) done in %.1fs",
            action_label(intervention_config), position, len(interventions), time.perf_counter() - started,
        )  # fmt: skip
    return pd.concat(tables, ignore_index=True), groupings


def metric_names(config: dict[str, Any]) -> list[str]:
    """Accuracy metrics, then beyond-accuracy metrics if configured."""
    evaluation = config["evaluation"]
    names = list(evaluation["metrics"]) + list((evaluation.get("beyond_accuracy") or {}).get("metrics", []))
    return [metric_name(m, k) for m in names for k in evaluation["k_values"]]


def run_intervention_experiment(config: dict[str, Any]) -> ExperimentResult:
    """Test-period analysis: per-action effects, state summaries, routing headroom."""
    interventions = expand_interventions(config["interventions"])
    validate_interventions(interventions, config["evaluation"]["k_values"])
    dataset, split, rngs = prepare_run(config)
    model = obtain_base_model(config, split.train)
    user_effects, groupings = evaluate_period(config, interventions, dataset, model, split)

    names = metric_names(config)
    bootstrap = config["analysis"]["bootstrap"]
    summary = summarize_effects(
        user_effects,
        metrics=names,
        groupings=groupings,
        n_resamples=bootstrap["n_resamples"],
        confidence=bootstrap["confidence"],
        rng=rngs["analysis"],
    )
    headroom = routing_headroom(user_effects, names, groupings)
    return ExperimentResult(user_effects=user_effects, summary=summary, headroom=headroom)


@dataclass(frozen=True)
class PeriodEffects:
    """Per-user effects for router training (validation) and evaluation (test)."""

    validation: pd.DataFrame
    test: pd.DataFrame


def validation_split(split: EvaluationSplit) -> EvaluationSplit:
    """Hold out each user's last *train* item: router labels never see test targets."""
    return leave_last_out(split.train, min_train_interactions=1)


EFFECT_SECTIONS = ("seed", "dataset", "model", "model_checkpoint", "interventions", "states", "evaluation")


def effects_cache_key(config: dict[str, Any]) -> str:
    """Hash of everything that changes per-user effects (routing settings excluded).
    A checkpoint file is identified by its size and modification time."""
    relevant = {section: config.get(section) for section in EFFECT_SECTIONS}
    relevant["neutral_threshold"] = config["analysis"]["neutral_threshold"]
    checkpoint = config.get("model_checkpoint")
    if checkpoint and Path(checkpoint).exists():
        stat = Path(checkpoint).stat()
        relevant["checkpoint_file"] = [stat.st_size, stat.st_mtime_ns]
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()


EFFECTS_KEY_FILENAME = "effects_key.txt"
EFFECTS_FILENAMES = {"validation": "validation_effects.csv", "test": "test_effects.csv"}


def load_or_compute_period_effects(config: dict[str, Any], cache_dir: Path, recompute: bool = False) -> PeriodEffects:
    """Per-user effects are the slow part; reuse them when nothing that affects them changed."""
    key = effects_cache_key(config)
    key_path = cache_dir / EFFECTS_KEY_FILENAME
    paths = {period: cache_dir / name for period, name in EFFECTS_FILENAMES.items()}
    cached = key_path.exists() and key_path.read_text().strip() == key and all(p.exists() for p in paths.values())
    if cached and not recompute:
        logger.info("reusing cached per-user effects from %s", cache_dir)
        return PeriodEffects(**{period: pd.read_csv(path) for period, path in paths.items()})
    effects = run_period_effects(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    effects.validation.to_csv(paths["validation"], index=False)
    effects.test.to_csv(paths["test"], index=False)
    key_path.write_text(key)
    return effects


def run_period_effects(config: dict[str, Any]) -> PeriodEffects:
    interventions = expand_interventions(config["interventions"])
    validate_interventions(interventions, config["evaluation"]["k_values"])
    dataset, split, _ = prepare_run(config)
    model = obtain_base_model(config, split.train)
    validation = validation_split(split)
    # A model fitted on all of split.train has seen the validation targets; refit it
    # on the shortened history unless it already held them out during fitting.
    validation_model = model if model.holds_out_last_item else build_model(config["model"]).fit(validation.train)
    logger.info("validation period: %d users, %d actions", len(validation.eval_users), len(interventions))
    valid_effects, _ = evaluate_period(config, interventions, dataset, validation_model, validation)
    logger.info("test period: %d users, %d actions", len(split.eval_users), len(interventions))
    test_effects, _ = evaluate_period(config, interventions, dataset, model, split)
    return PeriodEffects(validation=valid_effects, test=test_effects)
