"""Harm-aware routing of one beyond-accuracy re-ranker family.

A family is one re-ranker at several strengths (actions listed weakest first).
A damage model predicts, per (user, strength), whether applying it lowers the
label accuracy metric; the policy applies the strongest strength whose
predicted damage probability is <= tau, else keeps the base ranking. Sweeping
tau traces an objective-vs-accuracy frontier to compare with uniform strengths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from recint.analysis.effect import ACTION, USER_ID
from recint.routing.policy import NO_OP, realized
from recint.routing.router import _fit_or_constant, _proba


@dataclass(frozen=True)
class FamilyData:
    """Per-(user, action) arrays for one family; action axis ordered weakest first."""

    user_ids: np.ndarray
    actions: list[str]
    features: np.ndarray  # (n_users, n_actions, n_features)
    objective_gain: np.ndarray  # (n_users, n_actions), the family's own goal (higher = better)
    label_gain: np.ndarray  # accuracy metric whose decrease counts as damage
    report_gain: np.ndarray  # accuracy metric reported (e.g. HR@K -> Rescue/Damage)
    protect: np.ndarray  # (n_users,), base already correct on the report metric

    @property
    def damage(self) -> np.ndarray:
        return self.label_gain < 0


def build_family_data(
    user_effects: pd.DataFrame,
    actions: Sequence[str],
    user_features: Sequence[str],
    action_features: Sequence[str],
    objective_metric: str,
    label_metric: str,
    report_metric: str,
) -> FamilyData:
    first = user_effects[user_effects[ACTION] == actions[0]].set_index(USER_ID)

    def wide(column: str) -> np.ndarray:
        table = user_effects.pivot(index=USER_ID, columns=ACTION, values=column)
        return table.loc[first.index, list(actions)].to_numpy(dtype=np.float64)

    n_users, n_actions = len(first), len(actions)
    per_user = np.repeat(first[list(user_features)].to_numpy(dtype=np.float64)[:, None, :], n_actions, axis=1)
    per_action = [wide(column)[:, :, None] for column in action_features]
    strength = np.broadcast_to(np.linspace(0.0, 1.0, n_actions)[None, :, None], (n_users, n_actions, 1))
    return FamilyData(
        user_ids=first.index.to_numpy(),
        actions=list(actions),
        features=np.concatenate([per_user, *per_action, strength], axis=2),
        objective_gain=wide(f"{objective_metric}_gain"),
        label_gain=wide(f"{label_metric}_gain"),
        report_gain=wide(f"{report_metric}_gain"),
        protect=first[f"{report_metric}_base"].to_numpy() > 0,
    )


class DamageModel:
    """P(damage | user, strength); rows of base-correct users weigh `harm_weight`."""

    def __init__(self, classifier: str, harm_weight: float, seed: int) -> None:
        self.classifier = classifier
        self.harm_weight = harm_weight
        self.seed = seed

    def fit(self, data: FamilyData) -> DamageModel:
        n_users, n_actions, n_features = data.features.shape
        weight = np.repeat(np.where(data.protect, self.harm_weight, 1.0), n_actions)
        self._model = _fit_or_constant(
            self.classifier, self.seed, data.features.reshape(-1, n_features),
            data.damage.reshape(-1).astype(int), weight,
        )  # fmt: skip
        return self

    def predict(self, data: FamilyData) -> np.ndarray:
        n_users, n_actions, n_features = data.features.shape
        return _proba(self._model, data.features.reshape(-1, n_features), 2)[:, 1].reshape(n_users, n_actions)


def strongest_safe(p_damage: np.ndarray, threshold: float) -> np.ndarray:
    """Index of the strongest action with p_damage <= threshold, NO_OP if none."""
    allowed = p_damage <= threshold
    strongest = allowed.shape[1] - 1 - np.argmax(allowed[:, ::-1], axis=1)
    return np.where(allowed.any(axis=1), strongest, NO_OP)


def evaluate_choices(data: FamilyData, choices: np.ndarray) -> dict[str, float]:
    report = realized(data.report_gain, choices)
    return {
        "objective_gain": float(realized(data.objective_gain, choices).mean()),
        "report_gain": float(report.mean()),
        "label_gain": float(realized(data.label_gain, choices).mean()),
        "n_rescued": int((report > 0).sum()),
        "n_damaged": int((report < 0).sum()),
        "intervene_rate": float((choices != NO_OP).mean()),
    }


def uniform_choices(data: FamilyData, action: int) -> np.ndarray:
    return np.full(len(data.user_ids), action)


def gate_choices(values: np.ndarray, threshold: float, action: int) -> np.ndarray:
    """Apply `action` to users whose state value is >= threshold (e.g. uncertain users)."""
    return np.where(values >= threshold, action, NO_OP)


def match_threshold(objective_by_threshold: dict[float, float], target: float) -> float:
    """Most conservative threshold whose (validation) objective gain reaches `target`;
    the most permissive one if none does."""
    reaching = [t for t, objective in objective_by_threshold.items() if objective >= target]
    return min(reaching) if reaching else max(objective_by_threshold)
