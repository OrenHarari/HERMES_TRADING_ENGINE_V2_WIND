"""Prompt 2 / Step 7 - Paper Trading Validation Gate.

Two pure functions:

  evaluate_paper_validation(performance_report, *, edge_decay_alert,
      kill_switch_active, cost_model, account_equity, config)
      -> {"paper_validation_passed", "live_enabled", "reason", "details",
          "insufficient_regime_diversity"}

  check_live_trade_allowed(paper_validation_result, live_enabled,
      kill_switch_active)
      -> {"trade_allowed", "reason"}

Spec rules:
  - minimum 100 completed paper trades (configurable)
  - profit_factor >= configured minimum, default 1.20
  - max_drawdown <= configured maximum
  - no active edge_decay_alert
  - no active kill switch
  - valid cost model exists
  - valid account/equity state exists
  - report insufficient_regime_diversity when < 3 regimes; do NOT block by
    default; allow config to make it blocking

Note on lookahead/data-leakage: those checks are enforced upstream by the
data-contract (Step 0) and the lifecycle's no-future-data probe. The paper
gate trusts that those have already run. This step does not duplicate them.
"""

from hermes.safety.cost_model import CostModel
from hermes.utils.bounds import is_real_number

# ---- canonical reasons (do not rename) ---------------------------------
REASON_OK = ""
REASON_PAPER_VALIDATION_FAILED = "paper_validation_failed"
REASON_LIVE_NOT_ENABLED = "live_not_explicitly_enabled"
REASON_KILL_SWITCH_ACTIVE = "kill_switch_active"

# ---- detail strings (sub-reasons) --------------------------------------
DETAIL_OK = ""
DETAIL_INSUFFICIENT_PAPER_TRADES = "insufficient_paper_trades"
DETAIL_PROFIT_FACTOR_TOO_LOW = "profit_factor_too_low"
DETAIL_MAX_DRAWDOWN_TOO_HIGH = "max_drawdown_too_high"
DETAIL_EDGE_DECAY_ACTIVE = "edge_decay_active"
DETAIL_KILL_SWITCH_ACTIVE = "kill_switch_active"
DETAIL_MISSING_COST_MODEL = "missing_cost_model"
DETAIL_ACCOUNT_EQUITY_UNAVAILABLE = "account_equity_unavailable"
DETAIL_INSUFFICIENT_REGIME_DIVERSITY = "insufficient_regime_diversity"


class PaperGateConfig(object):
    """Configurable thresholds for paper-trading validation."""

    __slots__ = (
        "min_paper_trades",
        "min_profit_factor",
        "max_drawdown_max",
        "min_regime_diversity",
        "block_on_insufficient_regime_diversity",
    )

    def __init__(
        self,
        min_paper_trades=100,
        min_profit_factor=1.20,
        max_drawdown_max=float("inf"),
        min_regime_diversity=3,
        block_on_insufficient_regime_diversity=False,
    ):
        if not isinstance(min_paper_trades, int) or min_paper_trades < 0:
            raise ValueError("min_paper_trades must be int >= 0")
        if not is_real_number(min_profit_factor) or min_profit_factor < 0.0:
            raise ValueError("min_profit_factor must be a non-negative number")
        if not is_real_number(max_drawdown_max) or max_drawdown_max < 0.0:
            raise ValueError("max_drawdown_max must be a non-negative number")
        if not isinstance(min_regime_diversity, int) or min_regime_diversity < 0:
            raise ValueError("min_regime_diversity must be int >= 0")
        if not isinstance(block_on_insufficient_regime_diversity, bool):
            raise ValueError(
                "block_on_insufficient_regime_diversity must be bool"
            )
        self.min_paper_trades = min_paper_trades
        self.min_profit_factor = float(min_profit_factor)
        self.max_drawdown_max = float(max_drawdown_max)
        self.min_regime_diversity = min_regime_diversity
        self.block_on_insufficient_regime_diversity = (
            block_on_insufficient_regime_diversity
        )


def _result(passed, detail, insufficient_diversity):
    return {
        "paper_validation_passed": bool(passed),
        # Spec: passing paper does not automatically enable live. The
        # evaluator NEVER sets live_enabled=True by itself.
        "live_enabled": False,
        "reason": "" if passed else REASON_PAPER_VALIDATION_FAILED,
        "details": detail,
        "insufficient_regime_diversity": bool(insufficient_diversity),
    }


def evaluate_paper_validation(
    performance_report,
    edge_decay_alert,
    kill_switch_active,
    cost_model,
    account_equity,
    config=None,
):
    """Evaluate whether a system has passed paper-trading validation."""
    if config is None:
        config = PaperGateConfig()
    if not isinstance(performance_report, dict):
        raise ValueError("performance_report must be a dict")
    if not isinstance(edge_decay_alert, bool):
        raise ValueError("edge_decay_alert must be bool")
    if not isinstance(kill_switch_active, bool):
        raise ValueError("kill_switch_active must be bool")

    # --- regime diversity is reported regardless of pass/fail ----------
    tpr = performance_report.get("trades_per_regime", {}) or {}
    distinct_regimes = sum(1 for k, v in tpr.items() if v > 0)
    insufficient_diversity = distinct_regimes < config.min_regime_diversity

    # --- 1) cost model ---------------------------------------------------
    if cost_model is None or not isinstance(cost_model, CostModel):
        return _result(False, DETAIL_MISSING_COST_MODEL, insufficient_diversity)

    # --- 2) account equity (must be a positive number) ------------------
    if (
        account_equity is None
        or not is_real_number(account_equity)
        or account_equity <= 0.0
    ):
        return _result(
            False, DETAIL_ACCOUNT_EQUITY_UNAVAILABLE, insufficient_diversity
        )

    # --- 3) kill switch --------------------------------------------------
    if kill_switch_active:
        return _result(False, DETAIL_KILL_SWITCH_ACTIVE, insufficient_diversity)

    # --- 4) edge decay ---------------------------------------------------
    if edge_decay_alert:
        return _result(False, DETAIL_EDGE_DECAY_ACTIVE, insufficient_diversity)

    # --- 5) min paper trades --------------------------------------------
    trade_count = int(performance_report.get("trade_count", 0))
    if trade_count < config.min_paper_trades:
        return _result(
            False, DETAIL_INSUFFICIENT_PAPER_TRADES, insufficient_diversity
        )

    # --- 6) profit factor floor -----------------------------------------
    pf = performance_report.get("profit_factor", 0.0)
    # An infinite profit_factor (no losses) is acceptable -> only block
    # when finite & below floor.
    if is_real_number(pf) and pf != float("inf") and pf < config.min_profit_factor:
        return _result(
            False, DETAIL_PROFIT_FACTOR_TOO_LOW, insufficient_diversity
        )

    # --- 7) max drawdown cap --------------------------------------------
    mdd = performance_report.get("max_drawdown", 0.0)
    if is_real_number(mdd) and mdd > config.max_drawdown_max:
        return _result(
            False, DETAIL_MAX_DRAWDOWN_TOO_HIGH, insufficient_diversity
        )

    # --- 8) regime diversity (only blocks if config says so) ------------
    if (
        insufficient_diversity
        and config.block_on_insufficient_regime_diversity
    ):
        return _result(
            False, DETAIL_INSUFFICIENT_REGIME_DIVERSITY, insufficient_diversity
        )

    return _result(True, DETAIL_OK, insufficient_diversity)


def check_live_trade_allowed(
    paper_validation_result, live_enabled, kill_switch_active
):
    """Return {"trade_allowed", "reason"} for live execution.

    Real orders may execute only if ALL of:
      - paper_validation_result["paper_validation_passed"] is True
      - live_enabled is explicitly True
      - kill_switch_active is False
    """
    if not isinstance(paper_validation_result, dict):
        raise ValueError("paper_validation_result must be a dict")
    if "paper_validation_passed" not in paper_validation_result:
        raise ValueError(
            "paper_validation_result missing 'paper_validation_passed'"
        )
    if not isinstance(kill_switch_active, bool):
        raise ValueError("kill_switch_active must be bool")

    # 1) live_enabled must be explicitly True (per spec).
    if live_enabled is not True:
        return {"trade_allowed": False, "reason": REASON_LIVE_NOT_ENABLED}

    # 2) paper validation must have passed.
    if not paper_validation_result.get("paper_validation_passed", False):
        return {"trade_allowed": False, "reason": REASON_PAPER_VALIDATION_FAILED}

    # 3) kill switch must be inactive.
    if kill_switch_active:
        return {"trade_allowed": False, "reason": REASON_KILL_SWITCH_ACTIVE}

    return {"trade_allowed": True, "reason": ""}


__all__ = [
    "DETAIL_ACCOUNT_EQUITY_UNAVAILABLE",
    "DETAIL_EDGE_DECAY_ACTIVE",
    "DETAIL_INSUFFICIENT_PAPER_TRADES",
    "DETAIL_INSUFFICIENT_REGIME_DIVERSITY",
    "DETAIL_KILL_SWITCH_ACTIVE",
    "DETAIL_MAX_DRAWDOWN_TOO_HIGH",
    "DETAIL_MISSING_COST_MODEL",
    "DETAIL_OK",
    "DETAIL_PROFIT_FACTOR_TOO_LOW",
    "PaperGateConfig",
    "REASON_KILL_SWITCH_ACTIVE",
    "REASON_LIVE_NOT_ENABLED",
    "REASON_OK",
    "REASON_PAPER_VALIDATION_FAILED",
    "check_live_trade_allowed",
    "evaluate_paper_validation",
]
