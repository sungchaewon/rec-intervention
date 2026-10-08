"""Interventions applied on top of a (frozen) base recommender's output."""

from typing import Any

from recint.interventions.base import IdentityIntervention, Intervention
from recint.interventions.diversity import CalibrationReranker, MMRReranker
from recint.interventions.popularity import PopularityPenaltyReranker

INTERVENTION_REGISTRY: dict[str, type[Intervention]] = {
    "identity": IdentityIntervention,
    "popularity_penalty": PopularityPenaltyReranker,
    "mmr": MMRReranker,
    "calibration": CalibrationReranker,
}


def build_intervention(config: dict[str, Any]) -> Intervention:
    name = config["name"]
    if name not in INTERVENTION_REGISTRY:
        raise ValueError(
            f"Unknown intervention {name!r}; available: {sorted(INTERVENTION_REGISTRY)}"
        )
    return INTERVENTION_REGISTRY[name](**config.get("params", {}))


__all__ = [
    "INTERVENTION_REGISTRY",
    "CalibrationReranker",
    "IdentityIntervention",
    "Intervention",
    "MMRReranker",
    "PopularityPenaltyReranker",
    "build_intervention",
]
