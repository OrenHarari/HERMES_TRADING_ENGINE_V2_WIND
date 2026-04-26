"""Step 5 / Part 2 - Trade Eligibility Gate.

Returns {"trade_allowed": bool, "reason_if_blocked": str}.

Block reasons (deterministic priority order):
  1. low_confidence
  2. low_agreement
  3. regime_chop_disallowed
  4. volatility_too_low
  5. volatility_too_high

If all checks pass: {"trade_allowed": True, "reason_if_blocked": ""}.

This gate intentionally does NOT consult risk state or kill-switch state.
Risk guardrails are layered on top in `decision.make_decision`. This keeps the
eligibility gate a pure value-only function over signal+market context.
"""

from hermes.decision.config import DecisionConfig
from hermes.market import REGIME_CHOP, REGIME_VALUES
from hermes.utils.bounds import is_unit_interval

REASON_OK = ""
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_LOW_AGREEMENT = "low_agreement"
REASON_REGIME_CHOP = "regime_chop_disallowed"
REASON_VOL_LOW = "volatility_too_low"
REASON_VOL_HIGH = "volatility_too_high"


def check_eligibility(confidence, agreement, regime, volatility_score, config=None):
    """Pure value-only eligibility check.

    Returns: {"trade_allowed": bool, "reason_if_blocked": str}.
    Raises ValueError on invalid inputs.
    """
    if config is None:
        config = DecisionConfig()
    if not is_unit_interval(confidence):
        raise ValueError("confidence must be in [0,1]")
    if not is_unit_interval(agreement):
        raise ValueError("agreement must be in [0,1]")
    if not is_unit_interval(volatility_score):
        raise ValueError("volatility_score must be in [0,1]")
    if regime not in REGIME_VALUES:
        raise ValueError("invalid regime: {!r}".format(regime))

    if confidence < config.min_confidence:
        return {"trade_allowed": False, "reason_if_blocked": REASON_LOW_CONFIDENCE}
    if agreement < config.min_agreement:
        return {"trade_allowed": False, "reason_if_blocked": REASON_LOW_AGREEMENT}
    if regime == REGIME_CHOP and not config.allow_chop:
        return {"trade_allowed": False, "reason_if_blocked": REASON_REGIME_CHOP}
    if volatility_score < config.volatility_min:
        return {"trade_allowed": False, "reason_if_blocked": REASON_VOL_LOW}
    if volatility_score > config.volatility_max:
        return {"trade_allowed": False, "reason_if_blocked": REASON_VOL_HIGH}
    return {"trade_allowed": True, "reason_if_blocked": REASON_OK}
