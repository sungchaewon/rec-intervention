"""Observed intervention-effect analysis (WWW short-paper scope).

Consumes per-user base/intervention metrics and user states; does not choose
actions. A future routing module should consume `build_user_effect_table` output.
"""

from recint.analysis.bootstrap import bootstrap_mean_ci
from recint.analysis.effect import ALL_USERS_GROUP, build_user_effect_table, summarize_by_state

__all__ = [
    "ALL_USERS_GROUP",
    "bootstrap_mean_ci",
    "build_user_effect_table",
    "summarize_by_state",
]
