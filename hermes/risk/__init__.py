"""Risk layer (Prompt 1, Step 5 / Part 3).

Implements baseline guardrails and confidence-scaled position sizing.

Phase 2 (Step 5D) will add equity / stop-distance / pct-of-equity caps via a
sizing-safety wrapper next to this module - not a rewrite.
"""

from hermes.risk.config import RiskConfig
from hermes.risk.guardrails import (
    REASON_COOLDOWN,
    REASON_DAILY_LOSS,
    REASON_MAX_CONSECUTIVE_LOSSES,
    REASON_MAX_TRADES,
    RiskState,
)
from hermes.risk.sizing import compute_position_size
from hermes.risk.sizing_safety import (
    REASON_EQUITY_UNAVAILABLE,
    REASON_INVALID_STOP_DISTANCE,
    SizingSafetyConfig,
    safe_position_size,
)

__all__ = [
    "REASON_COOLDOWN",
    "REASON_DAILY_LOSS",
    "REASON_EQUITY_UNAVAILABLE",
    "REASON_INVALID_STOP_DISTANCE",
    "REASON_MAX_CONSECUTIVE_LOSSES",
    "REASON_MAX_TRADES",
    "RiskConfig",
    "RiskState",
    "SizingSafetyConfig",
    "compute_position_size",
    "safe_position_size",
]
