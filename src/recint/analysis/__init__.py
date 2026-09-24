"""Observed intervention-effect analysis.

Consumes per-user base/intervention metrics and user states; does not choose
actions. A router consumes `build_user_effect_table` output (states as
features, per-action gains as labels).
"""

from recint.analysis.bootstrap import bootstrap_mean_ci
from recint.analysis.effect import ALL_USERS, build_user_effect_table, group_column, summarize_effects
from recint.analysis.headroom import NO_OP_ACTION, routing_headroom

__all__ = [
    "ALL_USERS",
    "NO_OP_ACTION",
    "bootstrap_mean_ci",
    "build_user_effect_table",
    "group_column",
    "routing_headroom",
    "summarize_effects",
]
