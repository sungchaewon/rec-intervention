"""State-conditioned intervention routing (learn user state -> action).

Consumes per-user effect tables from the analysis stage: validation-period
effects train the router, test-period effects evaluate it.
"""

from recint.routing.data import RoutingData, build_routing_data
from recint.routing.policy import (
    NO_OP,
    best_uniform_choices,
    evaluate_policy,
    oracle_choices,
    uncertainty_gate_choices,
)
from recint.routing.router import GateSelectorRouter, SingleStageRouter

__all__ = [
    "NO_OP",
    "GateSelectorRouter",
    "RoutingData",
    "SingleStageRouter",
    "best_uniform_choices",
    "build_routing_data",
    "evaluate_policy",
    "oracle_choices",
    "uncertainty_gate_choices",
]
