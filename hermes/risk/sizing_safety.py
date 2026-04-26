"""Prompt 2 / Step 5D - Position Sizing Safety.

Stand-alone safe sizing wrapper. The Phase-1 `compute_position_size` in
`hermes.risk.sizing` is preserved unchanged. This module exposes a stricter
sizing function that:

  - blocks the trade when stop_distance <= 0 (invalid_stop_distance)
  - blocks the trade in paper/live mode if equity is missing
    (account_equity_unavailable)
  - applies ALL of the following caps:
      * absolute_risk_cap
      * equity * max_risk_per_trade
      * stop_loss distance (denominator)
      * confidence multiplier (clamped at confidence_multiplier_cap)
      * equity * max_position_pct_of_equity
      * available_capital

Formula (per Prompt 2 spec):

  risk_amount = min(equity * max_risk_per_trade, absolute_risk_cap)
  confidence_adjusted_risk = risk_amount * min(confidence, confidence_multiplier_cap)
  raw_size = confidence_adjusted_risk / stop_distance
  final_size = min(raw_size, equity * max_position_pct_of_equity, available_capital)

In backtest / legacy (system_mode is None or "backtest_mode") with no equity,
the equity-based components are skipped and risk_amount = absolute_risk_cap.
"""

from hermes.safety.data_contract import (
    SYSTEM_MODE_BACKTEST,
    SYSTEM_MODE_LIVE,
    SYSTEM_MODE_PAPER,
)
from hermes.utils.bounds import is_real_number, is_unit_interval

REASON_OK = ""
REASON_INVALID_STOP_DISTANCE = "invalid_stop_distance"
REASON_EQUITY_UNAVAILABLE = "account_equity_unavailable"

_PAPER_OR_LIVE = (SYSTEM_MODE_PAPER, SYSTEM_MODE_LIVE)


class SizingSafetyConfig(object):
    """Configurable risk caps for safe position sizing."""

    __slots__ = (
        "max_risk_per_trade",
        "max_position_pct_of_equity",
        "absolute_risk_cap",
        "confidence_multiplier_cap",
    )

    def __init__(
        self,
        max_risk_per_trade=0.01,
        max_position_pct_of_equity=0.20,
        absolute_risk_cap=10_000.0,
        confidence_multiplier_cap=1.0,
    ):
        if not is_unit_interval(max_risk_per_trade):
            raise ValueError("max_risk_per_trade must be in [0, 1]")
        if (
            not is_real_number(max_position_pct_of_equity)
            or max_position_pct_of_equity <= 0.0
        ):
            raise ValueError("max_position_pct_of_equity must be > 0")
        if not is_real_number(absolute_risk_cap) or absolute_risk_cap < 0.0:
            raise ValueError("absolute_risk_cap must be >= 0")
        if (
            not is_real_number(confidence_multiplier_cap)
            or confidence_multiplier_cap <= 0.0
        ):
            raise ValueError("confidence_multiplier_cap must be > 0")
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.max_position_pct_of_equity = float(max_position_pct_of_equity)
        self.absolute_risk_cap = float(absolute_risk_cap)
        self.confidence_multiplier_cap = float(confidence_multiplier_cap)


def _block(reason):
    return {
        "trade_allowed": False,
        "reason": reason,
        "position_size": 0.0,
        "details": {},
    }


def safe_position_size(
    equity,
    available_capital,
    confidence,
    stop_distance,
    system_mode=None,
    config=None,
):
    """Compute the safe position size and gate the trade if any required
    input is missing/invalid.

    Returns:
      {
        "trade_allowed": bool,
        "reason": str,
        "position_size": float,
        "details": {risk_amount, confidence_multiplier, raw_size_from_risk,
                    equity_pct_cap, available_capital_cap, applied_cap},
      }
    """
    if config is None:
        config = SizingSafetyConfig()

    # --- input validation -------------------------------------------------
    if not is_unit_interval(confidence):
        raise ValueError("confidence must be in [0, 1]")
    if not is_real_number(stop_distance):
        raise ValueError("stop_distance must be numeric")
    if equity is not None:
        if not is_real_number(equity) or equity < 0.0:
            raise ValueError("equity must be a non-negative number or None")
    if available_capital is not None:
        if not is_real_number(available_capital) or available_capital < 0.0:
            raise ValueError(
                "available_capital must be a non-negative number or None"
            )

    # --- gates ------------------------------------------------------------
    if stop_distance <= 0.0:
        return _block(REASON_INVALID_STOP_DISTANCE)
    if equity is None and system_mode in _PAPER_OR_LIVE:
        return _block(REASON_EQUITY_UNAVAILABLE)

    # --- formula ----------------------------------------------------------
    confidence_multiplier = min(
        float(confidence), float(config.confidence_multiplier_cap)
    )

    if equity is not None:
        risk_from_equity = float(equity) * config.max_risk_per_trade
        risk_amount = min(risk_from_equity, config.absolute_risk_cap)
    else:
        # Backtest / legacy without equity: only the absolute cap binds.
        risk_amount = config.absolute_risk_cap

    confidence_adjusted_risk = risk_amount * confidence_multiplier
    raw_size = confidence_adjusted_risk / float(stop_distance)

    if equity is not None:
        equity_pct_cap = float(equity) * config.max_position_pct_of_equity
    else:
        equity_pct_cap = float("inf")

    available_capital_cap = (
        float(available_capital) if available_capital is not None else float("inf")
    )

    candidates = [
        ("risk", raw_size),
        ("equity_pct", equity_pct_cap),
        ("available_capital", available_capital_cap),
    ]
    # Pick the smallest cap deterministically; ties resolve to the earliest
    # entry in the list above (so "risk" beats "equity_pct" beats
    # "available_capital" on equality).
    applied_cap = "risk"
    final_size = raw_size
    for name, value in candidates[1:]:
        if value < final_size:
            applied_cap = name
            final_size = value
    if final_size < 0.0:
        final_size = 0.0

    return {
        "trade_allowed": True,
        "reason": REASON_OK,
        "position_size": float(final_size),
        "details": {
            "risk_amount": float(risk_amount),
            "confidence_multiplier": float(confidence_multiplier),
            "raw_size_from_risk": float(raw_size),
            "equity_pct_cap": float(equity_pct_cap),
            "available_capital_cap": float(available_capital_cap),
            "applied_cap": applied_cap,
        },
    }
