"""Step 5 - Top-level make_decision().

Composes signal layer, market intelligence, confidence, eligibility gate,
risk guardrails, and baseline sizing into the canonical decision output:

  {
      "trade_allowed": bool,
      "confidence": float,
      "agreement": float,
      "regime": str,
      "position_size": float,
      "reason": str,
  }

Risk guardrail blocks take precedence over eligibility blocks (a kill-switch
in spirit, even before Phase 2's full kill switch). High confidence MUST NOT
override a risk block.

Phase 2 hooks (accepted but ignored in Phase 1):
  - system_mode (default "backtest")
  - safety_context (default None)
"""

from hermes.decision.confidence import compute_confidence_score
from hermes.decision.config import DecisionConfig
from hermes.decision.eligibility import REASON_OK, check_eligibility
from hermes.market import MarketIntelligenceConfig, assemble_intelligence
from hermes.orchestrator import build_signal_output
from hermes.risk.guardrails import RiskState
from hermes.risk.sizing import compute_position_size

REQUIRED_DECISION_KEYS = (
    "trade_allowed",
    "confidence",
    "agreement",
    "regime",
    "position_size",
    "reason",
)


def _blocked(reason, confidence, agreement, regime):
    return {
        "trade_allowed": False,
        "confidence": float(confidence),
        "agreement": float(agreement),
        "regime": str(regime),
        "position_size": 0.0,
        "reason": str(reason),
    }


def make_decision(
    raw_signal,
    candles,
    current_index,
    risk_state=None,
    now_ts=0,
    day_key="default",
    decision_config=None,
    market_config=None,
    regime_weights=None,
    confidence_weights=None,
    system_mode="backtest",
    safety_context=None,
):
    """Build the canonical decision dict.

    `raw_signal`: dict accepted by orchestrator.build_signal_output.
    `candles`, `current_index`: passed to market.assemble_intelligence.
    `risk_state`: optional RiskState; if None, a fresh empty state is used
                  (i.e. risk does not block on first call).
    `now_ts`, `day_key`: passed into the risk check (not used to read time).

    `system_mode` and `safety_context` are accepted for forward compatibility
    with Prompt 2; Phase 1 does not act on them.

    Returns a dict with EXACTLY the keys in REQUIRED_DECISION_KEYS.
    """
    # Forward-compat: accept but do not act on Prompt-2 args.
    _ = system_mode
    _ = safety_context

    if decision_config is None:
        decision_config = DecisionConfig()
    if market_config is None:
        market_config = MarketIntelligenceConfig()
    if risk_state is None:
        risk_state = RiskState()

    signal_output = build_signal_output(raw_signal)
    intelligence = assemble_intelligence(candles, current_index, market_config)
    confidence = compute_confidence_score(
        signal_output,
        intelligence,
        regime_weights=regime_weights,
        weights=confidence_weights,
    )

    # 1) Risk guardrail - takes precedence over eligibility (high confidence
    #    must never bypass risk caps).
    risk_check = risk_state.check(now_ts, day_key)
    if not risk_check["allowed"]:
        return _blocked(
            risk_check["reason"], confidence,
            signal_output["agreement"], intelligence["regime"],
        )

    # 2) Eligibility gate.
    elig = check_eligibility(
        confidence,
        signal_output["agreement"],
        intelligence["regime"],
        intelligence["volatility_score"],
        config=decision_config,
    )
    if not elig["trade_allowed"]:
        return _blocked(
            elig["reason_if_blocked"], confidence,
            signal_output["agreement"], intelligence["regime"],
        )

    # 3) Approved -> baseline position size.
    position_size = compute_position_size(
        decision_config.base_position_size,
        confidence,
        confidence_multiplier_cap=decision_config.confidence_multiplier_cap,
    )
    return {
        "trade_allowed": True,
        "confidence": float(confidence),
        "agreement": float(signal_output["agreement"]),
        "regime": str(intelligence["regime"]),
        "position_size": float(position_size),
        "reason": REASON_OK,
    }
