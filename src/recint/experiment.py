"""End-to-end observed-intervention-effect experiment.

data -> split -> base model -> intervention -> per-user metrics -> gain
     -> user state -> state-conditioned summary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from recint.analysis import build_user_effect_table, summarize_by_state
from recint.data import leave_last_out, load_interactions
from recint.interventions import build_intervention
from recint.metrics import compute_ranking_metrics, metric_name
from recint.models import build_model, mask_seen_items, top_k_items
from recint.states import assign_state_groups, build_state_extractor
from recint.utils import make_rngs, set_global_seed

RNG_STREAMS = ["data", "analysis"]


@dataclass(frozen=True)
class ExperimentResult:
    user_effects: pd.DataFrame
    state_summary: pd.DataFrame


def run_intervention_experiment(config: dict[str, Any]) -> ExperimentResult:
    validate_candidate_size(config)
    seed = config["seed"]
    set_global_seed(seed)
    rngs = make_rngs(seed, RNG_STREAMS)

    data = load_interactions(config["dataset"], rngs["data"])
    split = leave_last_out(data, config["dataset"]["split"]["min_train_interactions"])
    if len(split.eval_users) == 0:
        raise ValueError("No users have enough interactions for evaluation")

    model = build_model(config["model"]).fit(split.train)
    intervention = build_intervention(config["intervention"]).fit(split.train)

    # Dense (n_eval_users, n_items) scores: fine for toy data; batch for real catalogs.
    base_scores = model.score(split.eval_users)
    mask_seen_items(base_scores, split.train.user_item_matrix(), split.eval_users)
    intervened_scores = intervention.apply(split.eval_users, base_scores)

    evaluation = config["evaluation"]
    k_values = evaluation["k_values"]
    max_k = max(k_values)
    base_metrics = compute_ranking_metrics(
        top_k_items(base_scores, max_k), split.targets, evaluation["metrics"], k_values
    )
    intervention_metrics = compute_ranking_metrics(
        top_k_items(intervened_scores, max_k), split.targets, evaluation["metrics"], k_values
    )

    state_config = config["state"]
    state_values = build_state_extractor(state_config).compute(split.train)[split.eval_users]
    grouping = state_config["grouping"]
    state_groups = assign_state_groups(
        state_values, grouping["method"], grouping["cut_points"], grouping["labels"]
    )

    analysis = config["analysis"]
    user_effects = build_user_effect_table(
        user_ids=data.user_ids[split.eval_users],
        state_values=state_values,
        state_groups=state_groups,
        base_metrics=base_metrics,
        intervention_metrics=intervention_metrics,
        neutral_threshold=analysis["neutral_threshold"],
        intervention_name=config["intervention"]["name"],
    )
    metric_names = [metric_name(m, k) for m in evaluation["metrics"] for k in k_values]
    state_summary = summarize_by_state(
        user_effects,
        metrics=metric_names,
        group_order=grouping["labels"],
        n_resamples=analysis["bootstrap"]["n_resamples"],
        confidence=analysis["bootstrap"]["confidence"],
        rng=rngs["analysis"],
    )
    return ExperimentResult(user_effects=user_effects, state_summary=state_summary)


def validate_candidate_size(config: dict[str, Any]) -> None:
    """Rerankers can only return their candidates; top-K beyond that is ill-defined."""
    candidate_size = config["intervention"].get("params", {}).get("candidate_size")
    if candidate_size is not None and candidate_size < max(config["evaluation"]["k_values"]):
        raise ValueError(
            f"intervention candidate_size={candidate_size} must be >= max k "
            f"{max(config['evaluation']['k_values'])}"
        )

