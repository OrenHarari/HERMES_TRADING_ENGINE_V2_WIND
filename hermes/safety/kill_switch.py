"""Prompt 2 / Step 8 - Live Safety Kill Switch.

Stateful, deterministic, inspectable kill switch that:

  - blocks all trading once activated
  - cannot be bypassed by high confidence (no override hook in API)
  - preserves the original activation reason across re-trigger events
  - logs every activation, secondary trigger, reset attempt, and reset
    success/failure with timestamp + reason

Pure detector helpers:
  - detect_abnormal_slippage(actual, expected, max_slippage, multiplier)
  - detect_position_state_mismatch(engine vs broker state)
  - detect_corrupted_threshold_config(active_blob, candidate_blob)

Reset semantics:
  - live_mode: requires manual_reset=True AND every safety check pass
                (no auto-reset, ever)
  - backtest_mode / paper_mode: may auto-reset if
        consecutive_valid_candles >= config.min_consecutive_valid_candles
        AND minutes_since_last_execution_error >= config.min_minutes_since_execution_error
        AND no edge_decay_alert AND valid cost_model
  - any mode: cannot reset during edge_decay_alert
  - failed reset attempts are always logged; the kill switch remains active
"""

from hermes.safety.cost_model import CostModel
from hermes.safety.data_contract import (
    SYSTEM_MODE_BACKTEST,
    SYSTEM_MODE_LIVE,
    SYSTEM_MODE_PAPER,
    VALID_SYSTEM_MODES,
)
from hermes.utils.bounds import is_real_number

# THRESHOLD_BOUNDS is imported lazily inside the detector to avoid the
# import cycle: kill_switch -> learning.threshold_adapter -> learning.attribution
# -> decision.performance -> decision -> risk -> safety -> kill_switch.

# ---- canonical kill-switch reason strings (do not rename) -------------
KS_REASON_DAILY_LOSS = "daily_loss_limit_reached"
KS_REASON_MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses_reached"
KS_REASON_STALE_DATA = "stale_data_feed"
KS_REASON_DUPLICATE_CANDLE = "duplicate_candle"
KS_REASON_INVALID_CANDLE = "invalid_candle"
KS_REASON_ABNORMAL_SLIPPAGE = "abnormal_slippage"
KS_REASON_EXECUTION_ERROR = "execution_error"
KS_REASON_EDGE_DECAY = "edge_decay_alert"
KS_REASON_COST_MODEL_MISSING = "cost_model_missing"
KS_REASON_ACCOUNT_EQUITY_UNAVAILABLE = "account_equity_unavailable"
KS_REASON_POSITION_STATE_MISMATCH = "position_state_mismatch"
KS_REASON_UNKNOWN_SYSTEM_MODE = "unknown_system_mode"
KS_REASON_CORRUPTED_THRESHOLD_CONFIG = "corrupted_threshold_config"
KS_REASON_CORRUPTED_TRADE_STATE = "corrupted_trade_state"

KILL_SWITCH_REASONS = (
    KS_REASON_DAILY_LOSS,
    KS_REASON_MAX_CONSECUTIVE_LOSSES,
    KS_REASON_STALE_DATA,
    KS_REASON_DUPLICATE_CANDLE,
    KS_REASON_INVALID_CANDLE,
    KS_REASON_ABNORMAL_SLIPPAGE,
    KS_REASON_EXECUTION_ERROR,
    KS_REASON_EDGE_DECAY,
    KS_REASON_COST_MODEL_MISSING,
    KS_REASON_ACCOUNT_EQUITY_UNAVAILABLE,
    KS_REASON_POSITION_STATE_MISMATCH,
    KS_REASON_UNKNOWN_SYSTEM_MODE,
    KS_REASON_CORRUPTED_THRESHOLD_CONFIG,
    KS_REASON_CORRUPTED_TRADE_STATE,
)

REASON_KILL_SWITCH_ACTIVE = "kill_switch_active"


# ---- reset config ------------------------------------------------------
class ResetConfig(object):
    """Tunables for the auto-reset path (backtest/paper only)."""

    __slots__ = (
        "min_consecutive_valid_candles",
        "min_minutes_since_execution_error",
    )

    def __init__(
        self,
        min_consecutive_valid_candles=10,
        min_minutes_since_execution_error=30,
    ):
        if (
            not isinstance(min_consecutive_valid_candles, int)
            or min_consecutive_valid_candles < 0
        ):
            raise ValueError("min_consecutive_valid_candles must be int >= 0")
        if (
            not isinstance(min_minutes_since_execution_error, int)
            or min_minutes_since_execution_error < 0
        ):
            raise ValueError(
                "min_minutes_since_execution_error must be int >= 0"
            )
        self.min_consecutive_valid_candles = min_consecutive_valid_candles
        self.min_minutes_since_execution_error = (
            min_minutes_since_execution_error
        )


# ---- kill switch class -------------------------------------------------
class KillSwitch(object):
    """Stateful kill switch. Construction is deterministic; tests can compare
    state dicts directly.
    """

    __slots__ = ("_active", "_reason", "_activated_ts", "_log")

    def __init__(self):
        self._active = False
        self._reason = ""
        self._activated_ts = None
        self._log = []

    @property
    def active(self):
        return self._active

    @property
    def reason(self):
        return self._reason

    def get_log(self):
        return list(self._log)

    def state(self):
        return {
            "active": self._active,
            "reason": self._reason,
            "activated_ts": self._activated_ts,
            "log": list(self._log),
        }

    def activate(self, reason, now_ts):
        """Activate the kill switch with `reason`. If already active, the
        original reason is preserved; the new trigger is logged separately.
        """
        if reason not in KILL_SWITCH_REASONS:
            raise ValueError("unknown kill switch reason: {!r}".format(reason))
        if self._active:
            self._log.append({
                "event": "kill_switch_secondary_trigger",
                "reason": reason,
                "original_reason": self._reason,
                "timestamp": now_ts,
            })
            return
        self._active = True
        self._reason = reason
        self._activated_ts = now_ts
        self._log.append({
            "event": "kill_switch_activated",
            "reason": reason,
            "timestamp": now_ts,
        })

    def check_trade_allowed(self):
        """Return {trade_allowed, reason, kill_switch_reason}.

        Note: there is no `confidence` parameter or any other override hook;
        confidence cannot bypass the kill switch by construction.
        """
        if self._active:
            return {
                "trade_allowed": False,
                "reason": REASON_KILL_SWITCH_ACTIVE,
                "kill_switch_reason": self._reason,
            }
        return {
            "trade_allowed": True,
            "reason": "",
            "kill_switch_reason": "",
        }

    def attempt_reset(
        self,
        system_mode,
        now_ts,
        manual_reset,
        edge_decay_alert,
        execution_error_unresolved,
        consecutive_valid_candles,
        minutes_since_last_execution_error,
        cost_model,
        account_equity,
        config=None,
    ):
        """Attempt a safe reset. Returns
          {"reset_attempted": True,
           "reset_succeeded": bool,
           "reason": str}
        and always appends a log entry.
        """
        if config is None:
            config = ResetConfig()
        # Always log the attempt's outcome, regardless of branch.

        if system_mode not in VALID_SYSTEM_MODES:
            return self._record_reset_failure(
                now_ts, "unknown_system_mode"
            )
        if not isinstance(manual_reset, bool):
            raise ValueError("manual_reset must be bool")
        if not isinstance(edge_decay_alert, bool):
            raise ValueError("edge_decay_alert must be bool")
        if not isinstance(execution_error_unresolved, bool):
            raise ValueError("execution_error_unresolved must be bool")

        # Never reset during edge_decay_alert in any mode.
        if edge_decay_alert:
            return self._record_reset_failure(now_ts, "edge_decay_alert_active")

        # Cost model must be valid in any mode.
        if cost_model is None or not isinstance(cost_model, CostModel):
            return self._record_reset_failure(now_ts, "cost_model_missing")

        # Account equity must be valid in any mode.
        if (
            account_equity is None
            or not is_real_number(account_equity)
            or account_equity <= 0.0
        ):
            return self._record_reset_failure(
                now_ts, "account_equity_unavailable"
            )

        if execution_error_unresolved:
            return self._record_reset_failure(
                now_ts, "execution_error_unresolved"
            )

        if system_mode == SYSTEM_MODE_LIVE:
            # Spec: live mode must NOT auto-reset.
            if not manual_reset:
                return self._record_reset_failure(
                    now_ts, "live_mode_requires_manual_reset"
                )
            # All safety checks already passed above -> reset is allowed.
            return self._record_reset_success(now_ts, "manual_reset_live")

        # Backtest / paper mode: auto-reset path.
        if (
            not isinstance(consecutive_valid_candles, int)
            or consecutive_valid_candles < 0
        ):
            raise ValueError("consecutive_valid_candles must be int >= 0")
        if minutes_since_last_execution_error is not None:
            if (
                not is_real_number(minutes_since_last_execution_error)
                or minutes_since_last_execution_error < 0.0
            ):
                raise ValueError(
                    "minutes_since_last_execution_error must be a "
                    "non-negative number or None"
                )
        if consecutive_valid_candles < config.min_consecutive_valid_candles:
            return self._record_reset_failure(
                now_ts, "insufficient_valid_candle_streak"
            )
        if (
            minutes_since_last_execution_error is None
            or minutes_since_last_execution_error
            < config.min_minutes_since_execution_error
        ):
            return self._record_reset_failure(
                now_ts, "execution_error_quiet_period_not_met"
            )
        return self._record_reset_success(
            now_ts, "auto_reset_{!s}".format(system_mode)
        )

    # ----- internal helpers ----------------------------------------------
    def _record_reset_success(self, now_ts, why):
        prior_reason = self._reason
        self._log.append({
            "event": "kill_switch_reset",
            "previous_reason": prior_reason,
            "outcome": why,
            "timestamp": now_ts,
        })
        self._active = False
        self._reason = ""
        self._activated_ts = None
        return {
            "reset_attempted": True,
            "reset_succeeded": True,
            "reason": why,
        }

    def _record_reset_failure(self, now_ts, why):
        self._log.append({
            "event": "kill_switch_reset_failed",
            "reason_kill_switch_remains": self._reason,
            "failure_reason": why,
            "timestamp": now_ts,
        })
        return {
            "reset_attempted": True,
            "reset_succeeded": False,
            "reason": why,
        }


# ---- pure detector helpers ---------------------------------------------
def detect_abnormal_slippage(
    actual_slippage, expected_slippage, max_slippage, slippage_multiplier
):
    """Return True iff actual exceeds either:
      - max_slippage (absolute cap), OR
      - expected_slippage * slippage_multiplier
    """
    for name, val in (
        ("actual_slippage", actual_slippage),
        ("expected_slippage", expected_slippage),
        ("max_slippage", max_slippage),
        ("slippage_multiplier", slippage_multiplier),
    ):
        if not is_real_number(val) or val < 0.0:
            raise ValueError("{!s} must be a non-negative number".format(name))
    if actual_slippage > max_slippage:
        return True
    if actual_slippage > expected_slippage * slippage_multiplier:
        return True
    return False


def detect_position_state_mismatch(
    engine_open,
    broker_open,
    engine_size,
    broker_size,
    engine_entry_price,
    broker_position_count,
    max_open_positions=1,
    material_size_diff_pct=0.05,
):
    """Return {"mismatch": bool, "reason": str}."""
    if not isinstance(engine_open, bool):
        raise ValueError("engine_open must be bool")
    if not isinstance(broker_open, bool):
        raise ValueError("broker_open must be bool")
    if (
        not isinstance(broker_position_count, int)
        or broker_position_count < 0
    ):
        raise ValueError("broker_position_count must be int >= 0")
    if engine_open != broker_open:
        return {
            "mismatch": True,
            "reason": "engine_broker_state_disagree",
        }
    if broker_position_count > max_open_positions:
        return {"mismatch": True, "reason": "too_many_open_positions"}
    if engine_open:
        if engine_entry_price is None:
            return {
                "mismatch": True,
                "reason": "missing_entry_price_for_open_position",
            }
        if engine_size <= 0.0 or broker_size <= 0.0:
            return {
                "mismatch": True,
                "reason": "non_positive_size_on_open_position",
            }
        denom = max(engine_size, broker_size)
        if denom > 0.0:
            diff = abs(engine_size - broker_size) / denom
            if diff > material_size_diff_pct:
                return {
                    "mismatch": True,
                    "reason": "material_size_difference",
                }
    return {"mismatch": False, "reason": ""}


_REQUIRED_THRESHOLD_KEYS = ("min_confidence", "allow_chop")
_SUPPORTED_SCHEMA_VERSIONS = (1,)


def _key_corrupted(blob):
    """Return (corrupted, reason) for a single thresholds blob."""
    # Lazy import to avoid circular dependency at module load time.
    from hermes.learning.threshold_adapter import THRESHOLD_BOUNDS

    if not isinstance(blob, dict):
        return True, "blob_not_dict"
    if "schema_version" not in blob:
        return True, "missing_schema_version"
    if blob["schema_version"] not in _SUPPORTED_SCHEMA_VERSIONS:
        return True, "unsupported_schema_version"
    for k in _REQUIRED_THRESHOLD_KEYS:
        if k not in blob:
            return True, "missing_required_key"
    # min_confidence numeric + within bounds.
    mc = blob.get("min_confidence")
    if (
        isinstance(mc, bool)
        or not isinstance(mc, (int, float))
        or mc != mc  # NaN
    ):
        return True, "non_numeric_min_confidence"
    lo, hi = THRESHOLD_BOUNDS["min_confidence"]
    if mc < lo or mc > hi:
        return True, "min_confidence_out_of_bounds"
    if not isinstance(blob.get("allow_chop"), bool):
        return True, "non_bool_allow_chop"
    return False, ""


def detect_corrupted_threshold_config(active_blob, candidate_blob=None):
    """Return {"corrupted": bool, "reason": str}.

    Active blob is required; candidate blob is optional. If candidate is
    present, its schema (set of non-schema_version keys) must match active's.
    """
    corrupted, reason = _key_corrupted(active_blob)
    if corrupted:
        return {"corrupted": True, "reason": "active_" + reason}
    if candidate_blob is not None:
        ccorrupted, creason = _key_corrupted(candidate_blob)
        if ccorrupted:
            return {"corrupted": True, "reason": "candidate_" + creason}
        # Schema parity check: same key set (excluding schema_version).
        active_keys = {k for k in active_blob.keys() if k != "schema_version"}
        cand_keys = {k for k in candidate_blob.keys() if k != "schema_version"}
        if active_keys != cand_keys:
            return {"corrupted": True, "reason": "schema_mismatch"}
    return {"corrupted": False, "reason": ""}


__all__ = [
    "KILL_SWITCH_REASONS",
    "KS_REASON_ABNORMAL_SLIPPAGE",
    "KS_REASON_ACCOUNT_EQUITY_UNAVAILABLE",
    "KS_REASON_CORRUPTED_THRESHOLD_CONFIG",
    "KS_REASON_CORRUPTED_TRADE_STATE",
    "KS_REASON_COST_MODEL_MISSING",
    "KS_REASON_DAILY_LOSS",
    "KS_REASON_DUPLICATE_CANDLE",
    "KS_REASON_EDGE_DECAY",
    "KS_REASON_EXECUTION_ERROR",
    "KS_REASON_INVALID_CANDLE",
    "KS_REASON_MAX_CONSECUTIVE_LOSSES",
    "KS_REASON_POSITION_STATE_MISMATCH",
    "KS_REASON_STALE_DATA",
    "KS_REASON_UNKNOWN_SYSTEM_MODE",
    "KillSwitch",
    "REASON_KILL_SWITCH_ACTIVE",
    "ResetConfig",
    "detect_abnormal_slippage",
    "detect_corrupted_threshold_config",
    "detect_position_state_mismatch",
]
