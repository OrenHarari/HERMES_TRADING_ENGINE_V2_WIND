"""Integration Layer - safe_make_decision.

A thin, deterministic wrapper around `hermes.decision.make_decision` that
runs the Prompt-2 safety pipeline around the Prompt-1 core. The Phase-1
core is NOT modified; this module is purely additive.

Pipeline (first-blocker wins):

  1. validate_system_mode               -> "invalid_system_mode"
  2. validate_market_data               -> "invalid_market_data" / "stale_market_data"
  3. KillSwitch.check_trade_allowed     -> "kill_switch_active"
  4. (paper/live) check_cost_model_for_live  -> "missing_cost_model"
  5. (live)  check_live_trade_allowed   -> "paper_validation_failed"
                                          / "live_not_explicitly_enabled"
  6. core: make_decision(...)           -> may itself return a Phase-1 reason
  7. (paper/live) safe_position_size overlay
                                        -> "invalid_stop_distance"
                                        / "account_equity_unavailable"
  8. assemble final output

Output is ALWAYS the same key set (blocked or approved):

  Phase-1 mandatory (unchanged shape):
    trade_allowed, confidence, agreement, regime, position_size, reason

  Phase-2 additive (explicitly listed in PROMPT_2 backward-compat block):
    system_mode, kill_switch_active, kill_switch_reason,
    cost_model_applied, live_enabled, paper_validation_passed,
    blocked_by_stage

This module is the ONLY place these 13 keys are assembled.
"""

from hermes.decision.decision import make_decision
from hermes.risk.sizing_safety import (
    REASON_EQUITY_UNAVAILABLE,
    REASON_INVALID_STOP_DISTANCE,
    safe_position_size,
)
from hermes.safety.cost_model import (
    REASON_MISSING_COST_MODEL,
    CostModel,
    check_cost_model_for_live,
)
from hermes.safety.data_contract import (
    SYSTEM_MODE_BACKTEST,
    SYSTEM_MODE_LIVE,
    SYSTEM_MODE_PAPER,
    validate_market_data,
)
from hermes.safety.kill_switch import KillSwitch
from hermes.safety.paper_gate import (
    REASON_LIVE_NOT_ENABLED,
    REASON_PAPER_VALIDATION_FAILED,
    check_live_trade_allowed,
)

# ---- canonical output keys (do not rename) ----------------------------
PHASE1_OUTPUT_KEYS = (
    "trade_allowed",
    "confidence",
    "agreement",
    "regime",
    "position_size",
    "reason",
)

PHASE2_ADDITIVE_KEYS = (
    "system_mode",
    "kill_switch_active",
    "kill_switch_reason",
    "cost_model_applied",
    "live_enabled",
    "paper_validation_passed",
    "blocked_by_stage",
)

INTEGRATION_OUTPUT_KEYS = PHASE1_OUTPUT_KEYS + PHASE2_ADDITIVE_KEYS

# Stage labels stored in 'blocked_by_stage'.
STAGE_NONE = ""
STAGE_DATA_CONTRACT = "data_contract"
STAGE_KILL_SWITCH = "kill_switch"
STAGE_COST_MODEL = "cost_model"
STAGE_LIVE_GATE = "live_gate"
STAGE_CORE = "core"
STAGE_SIZING_SAFETY = "sizing_safety"

_PAPER_OR_LIVE = (SYSTEM_MODE_PAPER, SYSTEM_MODE_LIVE)


def _assemble_output(
    *,
    trade_allowed,
    confidence,
    agreement,
    regime,
    position_size,
    reason,
    system_mode,
    kill_switch_active,
    kill_switch_reason,
    cost_model_applied,
    live_enabled,
    paper_validation_passed,
    blocked_by_stage,
):
    """Single source of truth for the integrated output dict.

    Every return path in `safe_make_decision` goes through this builder so
    the key set is guaranteed identical across blocked and approved paths.
    """
    return {
        "trade_allowed": bool(trade_allowed),
        "confidence": float(confidence),
        "agreement": float(agreement),
        "regime": str(regime),
        "position_size": float(position_size) if position_size is not None else 0.0,
        "reason": str(reason),
        "system_mode": str(system_mode) if system_mode is not None else "",
        "kill_switch_active": bool(kill_switch_active),
        "kill_switch_reason": str(kill_switch_reason),
        "cost_model_applied": bool(cost_model_applied),
        "live_enabled": bool(live_enabled),
        "paper_validation_passed": bool(paper_validation_passed),
        "blocked_by_stage": str(blocked_by_stage),
    }


def _blocked_output_pre_core(
    reason,
    blocked_by_stage,
    *,
    system_mode,
    kill_switch_active,
    kill_switch_reason,
    cost_model_applied,
    live_enabled,
    paper_validation_passed,
):
    """Build a blocked output dict for stages that run BEFORE the core.

    Confidence / agreement / regime are unknown at these stages, so they are
    populated with stable sentinel defaults (0.0 / 0.0 / "").
    """
    return _assemble_output(
        trade_allowed=False,
        confidence=0.0,
        agreement=0.0,
        regime="",
        position_size=0.0,
        reason=reason,
        system_mode=system_mode,
        kill_switch_active=kill_switch_active,
        kill_switch_reason=kill_switch_reason,
        cost_model_applied=cost_model_applied,
        live_enabled=live_enabled,
        paper_validation_passed=paper_validation_passed,
        blocked_by_stage=blocked_by_stage,
    )


def safe_make_decision(
    raw_signal,
    candles,
    current_index,
    *,
    # ---- safety context ------------------------------------------------
    system_mode=None,
    now_ts=None,
    max_staleness_seconds=None,
    kill_switch=None,
    cost_model=None,
    account_equity=None,
    available_capital=None,
    stop_distance=None,
    paper_validation_result=None,
    live_enabled=False,
    safe_sizing_config=None,
    # ---- Phase-1 passthrough ------------------------------------------
    risk_state=None,
    day_key="default",
    decision_config=None,
    market_config=None,
    regime_weights=None,
    confidence_weights=None,
):
    """Run the Prompt-2 safety pipeline around `make_decision`.

    All safety inputs are kwargs-only; in backtest / legacy (`system_mode=None`
    or `system_mode="backtest_mode"`) most of them are unused, so callers
    who do not opt into safety still get a deterministic result identical
    to `make_decision` in shape (plus the additive keys).

    Returns a dict with EXACTLY the keys in `INTEGRATION_OUTPUT_KEYS`.
    """
    # Validate kill_switch type early so misuse is loud.
    if kill_switch is not None and not isinstance(kill_switch, KillSwitch):
        raise ValueError("kill_switch must be a KillSwitch instance or None")
    if cost_model is not None and not isinstance(cost_model, CostModel):
        raise ValueError("cost_model must be a CostModel instance or None")
    if not isinstance(live_enabled, bool):
        raise ValueError("live_enabled must be a bool")

    # Snapshot kill-switch state once for output reporting.
    ks_active = bool(kill_switch.active) if kill_switch is not None else False
    ks_reason = (
        str(kill_switch.reason) if (kill_switch is not None and ks_active) else ""
    )
    cm_present = isinstance(cost_model, CostModel)
    paper_passed = bool(
        paper_validation_result.get("paper_validation_passed", False)
    ) if isinstance(paper_validation_result, dict) else False

    # ---------- Stage 1+2: data contract ------------------------------
    data_check = validate_market_data(
        candles,
        current_index,
        system_mode=system_mode,
        now_ts=now_ts,
        max_staleness_seconds=max_staleness_seconds,
    )
    if not data_check["trade_allowed"]:
        return _blocked_output_pre_core(
            reason=data_check["reason"],
            blocked_by_stage=STAGE_DATA_CONTRACT,
            system_mode=system_mode,
            kill_switch_active=ks_active,
            kill_switch_reason=ks_reason,
            cost_model_applied=cm_present,
            live_enabled=live_enabled,
            paper_validation_passed=paper_passed,
        )

    # ---------- Stage 3: kill switch ----------------------------------
    if kill_switch is not None:
        ks_check = kill_switch.check_trade_allowed()
        if not ks_check["trade_allowed"]:
            return _blocked_output_pre_core(
                reason=ks_check["reason"],
                blocked_by_stage=STAGE_KILL_SWITCH,
                system_mode=system_mode,
                kill_switch_active=True,
                kill_switch_reason=ks_check["kill_switch_reason"],
                cost_model_applied=cm_present,
                live_enabled=live_enabled,
                paper_validation_passed=paper_passed,
            )

    # ---------- Stage 4: cost-model presence (paper/live only) -------
    if system_mode in _PAPER_OR_LIVE:
        cm_check = check_cost_model_for_live(cost_model)
        if not cm_check["trade_allowed"]:
            return _blocked_output_pre_core(
                reason=cm_check["reason"],
                blocked_by_stage=STAGE_COST_MODEL,
                system_mode=system_mode,
                kill_switch_active=ks_active,
                kill_switch_reason=ks_reason,
                cost_model_applied=False,
                live_enabled=live_enabled,
                paper_validation_passed=paper_passed,
            )

    # ---------- Stage 5: live trade gate (live only) -----------------
    if system_mode == SYSTEM_MODE_LIVE:
        # Use the supplied paper_validation_result; if absent, build a
        # synthetic "failed" result so the live gate's reason path runs
        # uniformly.
        pv = (
            paper_validation_result
            if isinstance(paper_validation_result, dict)
            else {"paper_validation_passed": False}
        )
        live_check = check_live_trade_allowed(
            paper_validation_result=pv,
            live_enabled=live_enabled,
            kill_switch_active=ks_active,
        )
        if not live_check["trade_allowed"]:
            return _blocked_output_pre_core(
                reason=live_check["reason"],
                blocked_by_stage=STAGE_LIVE_GATE,
                system_mode=system_mode,
                kill_switch_active=ks_active,
                kill_switch_reason=ks_reason,
                cost_model_applied=cm_present,
                live_enabled=live_enabled,
                paper_validation_passed=paper_passed,
            )

    # ---------- Stage 6: core engine (UNCHANGED) ---------------------
    # Phase-1 make_decision still ignores system_mode/safety_context; the
    # wrapper has already applied the canonical safety reading above.
    core = make_decision(
        raw_signal,
        candles,
        current_index,
        risk_state=risk_state,
        now_ts=now_ts if now_ts is not None else 0,
        day_key=day_key,
        decision_config=decision_config,
        market_config=market_config,
        regime_weights=regime_weights,
        confidence_weights=confidence_weights,
        system_mode=(system_mode if system_mode is not None else "backtest"),
        safety_context=None,
    )

    if not core["trade_allowed"]:
        return _assemble_output(
            trade_allowed=False,
            confidence=core["confidence"],
            agreement=core["agreement"],
            regime=core["regime"],
            position_size=0.0,
            reason=core["reason"],
            system_mode=system_mode,
            kill_switch_active=ks_active,
            kill_switch_reason=ks_reason,
            cost_model_applied=cm_present,
            live_enabled=live_enabled,
            paper_validation_passed=paper_passed,
            blocked_by_stage=STAGE_CORE,
        )

    # ---------- Stage 7: safe-position-size overlay (paper/live) -----
    final_position_size = float(core["position_size"])
    if system_mode in _PAPER_OR_LIVE:
        sps = safe_position_size(
            equity=account_equity,
            available_capital=available_capital,
            confidence=core["confidence"],
            stop_distance=(
                stop_distance if stop_distance is not None else 0.0
            ),
            system_mode=system_mode,
            config=safe_sizing_config,
        )
        if not sps["trade_allowed"]:
            return _assemble_output(
                trade_allowed=False,
                confidence=core["confidence"],
                agreement=core["agreement"],
                regime=core["regime"],
                position_size=0.0,
                reason=sps["reason"],
                system_mode=system_mode,
                kill_switch_active=ks_active,
                kill_switch_reason=ks_reason,
                cost_model_applied=cm_present,
                live_enabled=live_enabled,
                paper_validation_passed=paper_passed,
                blocked_by_stage=STAGE_SIZING_SAFETY,
            )
        # Safe sizing may only reduce, never increase, the core's size.
        if sps["position_size"] < final_position_size:
            final_position_size = float(sps["position_size"])

    # ---------- Stage 8: approved -----------------------------------
    return _assemble_output(
        trade_allowed=True,
        confidence=core["confidence"],
        agreement=core["agreement"],
        regime=core["regime"],
        position_size=final_position_size,
        reason=core["reason"],  # "" on approval
        system_mode=system_mode,
        kill_switch_active=ks_active,
        kill_switch_reason=ks_reason,
        cost_model_applied=cm_present,
        live_enabled=live_enabled,
        paper_validation_passed=paper_passed,
        blocked_by_stage=STAGE_NONE,
    )


__all__ = [
    "INTEGRATION_OUTPUT_KEYS",
    "PHASE1_OUTPUT_KEYS",
    "PHASE2_ADDITIVE_KEYS",
    "STAGE_CORE",
    "STAGE_COST_MODEL",
    "STAGE_DATA_CONTRACT",
    "STAGE_KILL_SWITCH",
    "STAGE_LIVE_GATE",
    "STAGE_NONE",
    "STAGE_SIZING_SAFETY",
    "safe_make_decision",
]
