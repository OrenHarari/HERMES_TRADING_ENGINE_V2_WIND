"""Prompt 2 / Step 6D - Out-of-Sample Promotion Gate.

Pure validation comparator and a window-split helper. The output of
`evaluate_oos_promotion` is the `validation_result` dict consumed by
`hermes.learning.candidate_thresholds.promote_candidate`.

Spec rules implemented (in deterministic priority order):
  1. validation window must be strictly later than discovery window
  2. validation trade count must be >= min_validation_trades
  3. candidate trade count must keep at least min_trade_count_ratio of active
  4. profit_factor must NOT deteriorate beyond profit_factor_min_ratio
  5. max_drawdown must NOT increase beyond max_drawdown_max_increase_pct
  6. stability_score must be >= min_stability_score
  7. win_rate alone is insufficient if either drawdown or profit_factor
     materially worsens (already covered by 4 + 5 above)

If all checks pass: {"validation_passed": True, "reason": "", "details": {...}}.

Window-split helper:
  split_trades_into_oos_windows(trades, validation_months=1) ->
    {"discovery_trades", "validation_trades",
     "discovery_window", "validation_window"}

Trades are bucketed by the 'date' field's leading 'YYYY-MM' month key, the
same convention used by walk_forward.
"""

from hermes.utils.bounds import is_real_number

# ---- canonical reasons (do not rename) ---------------------------------
REASON_OK = ""
REASON_VALIDATION_NOT_AFTER_DISCOVERY = "validation_not_after_discovery"
REASON_INSUFFICIENT_VALIDATION_TRADES = "insufficient_validation_trades"
REASON_TRADE_COUNT_REDUCED = "candidate_reduces_trade_count"
REASON_PROFIT_FACTOR_DETERIORATED = "profit_factor_deteriorated"
REASON_MAX_DRAWDOWN_INCREASED = "max_drawdown_increased"
REASON_STABILITY_TOO_LOW = "stability_score_too_low"
REASON_WIN_RATE_BUT_RISK_WORSENS = "win_rate_improves_but_risk_worsens"


class OOSValidationConfig(object):
    """Tunable thresholds for the OOS promotion gate."""

    __slots__ = (
        "min_validation_trades",
        "min_trade_count_ratio",
        "profit_factor_min_ratio",
        "max_drawdown_max_increase_pct",
        "min_stability_score",
    )

    def __init__(
        self,
        min_validation_trades=30,
        min_trade_count_ratio=0.5,
        profit_factor_min_ratio=0.95,
        max_drawdown_max_increase_pct=0.20,
        min_stability_score=0.30,
    ):
        if not isinstance(min_validation_trades, int) or min_validation_trades < 0:
            raise ValueError("min_validation_trades must be int >= 0")
        for name, val in (
            ("min_trade_count_ratio", min_trade_count_ratio),
            ("profit_factor_min_ratio", profit_factor_min_ratio),
            ("max_drawdown_max_increase_pct", max_drawdown_max_increase_pct),
            ("min_stability_score", min_stability_score),
        ):
            if not is_real_number(val) or val < 0.0:
                raise ValueError("{!s} must be a non-negative number".format(name))
        self.min_validation_trades = min_validation_trades
        self.min_trade_count_ratio = float(min_trade_count_ratio)
        self.profit_factor_min_ratio = float(profit_factor_min_ratio)
        self.max_drawdown_max_increase_pct = float(max_drawdown_max_increase_pct)
        self.min_stability_score = float(min_stability_score)


def _strictly_after(validation_window, discovery_window):
    """Validation window must start strictly AFTER discovery window ends.

    Both windows are (start, end) tuples of comparable values (e.g. month
    strings 'YYYY-MM' or numeric timestamps).
    """
    return validation_window[0] > discovery_window[1]


def _window_ok(window):
    return (
        isinstance(window, tuple)
        and len(window) == 2
        and window[0] is not None
        and window[1] is not None
    )


def _result(passed, reason, details):
    return {
        "validation_passed": bool(passed),
        "reason": reason,
        "details": dict(details),
    }


def evaluate_oos_promotion(
    discovery_window,
    validation_window,
    active_validation_report,
    candidate_validation_report,
    config=None,
):
    """Pure comparator. Returns
      {"validation_passed": bool, "reason": str, "details": dict}.

    Both reports must be performance-report-shaped dicts produced on the SAME
    validation window (one under active thresholds, one under candidate
    thresholds).
    """
    if config is None:
        config = OOSValidationConfig()
    if not _window_ok(discovery_window):
        raise ValueError("discovery_window must be a (start, end) tuple")
    if not _window_ok(validation_window):
        raise ValueError("validation_window must be a (start, end) tuple")
    if not isinstance(active_validation_report, dict):
        raise ValueError("active_validation_report must be a dict")
    if not isinstance(candidate_validation_report, dict):
        raise ValueError("candidate_validation_report must be a dict")

    details = {
        "discovery_window": list(discovery_window),
        "validation_window": list(validation_window),
        "active_trade_count": active_validation_report.get("trade_count", 0),
        "candidate_trade_count": candidate_validation_report.get(
            "trade_count", 0
        ),
        "active_profit_factor": active_validation_report.get(
            "profit_factor", 0.0
        ),
        "candidate_profit_factor": candidate_validation_report.get(
            "profit_factor", 0.0
        ),
        "active_max_drawdown": active_validation_report.get(
            "max_drawdown", 0.0
        ),
        "candidate_max_drawdown": candidate_validation_report.get(
            "max_drawdown", 0.0
        ),
        "active_stability": active_validation_report.get(
            "stability_score", 0.0
        ),
        "candidate_stability": candidate_validation_report.get(
            "stability_score", 0.0
        ),
        "active_win_rate": active_validation_report.get("win_rate", 0.0),
        "candidate_win_rate": candidate_validation_report.get(
            "win_rate", 0.0
        ),
    }

    # 1) window order
    if not _strictly_after(validation_window, discovery_window):
        return _result(False, REASON_VALIDATION_NOT_AFTER_DISCOVERY, details)

    # 2) min validation trades (use the candidate's count -- it's the system
    # that would actually trade)
    cand_count = int(details["candidate_trade_count"])
    if cand_count < config.min_validation_trades:
        return _result(False, REASON_INSUFFICIENT_VALIDATION_TRADES, details)

    # 3) candidate trade count vs active
    active_count = int(details["active_trade_count"])
    if active_count > 0:
        ratio = float(cand_count) / float(active_count)
        details["trade_count_ratio"] = ratio
        if ratio < config.min_trade_count_ratio:
            return _result(False, REASON_TRADE_COUNT_REDUCED, details)

    # 4) profit factor must not deteriorate beyond ratio
    active_pf = float(details["active_profit_factor"])
    cand_pf = float(details["candidate_profit_factor"])
    if active_pf > 0.0:
        # cand_pf >= active_pf * profit_factor_min_ratio
        threshold = active_pf * config.profit_factor_min_ratio
        details["profit_factor_threshold"] = threshold
        if cand_pf < threshold:
            return _result(False, REASON_PROFIT_FACTOR_DETERIORATED, details)

    # 5) max drawdown must not increase beyond pct
    active_mdd = float(details["active_max_drawdown"])
    cand_mdd = float(details["candidate_max_drawdown"])
    if active_mdd > 0.0:
        max_allowed = active_mdd * (1.0 + config.max_drawdown_max_increase_pct)
        details["max_drawdown_threshold"] = max_allowed
        if cand_mdd > max_allowed:
            return _result(False, REASON_MAX_DRAWDOWN_INCREASED, details)
    elif cand_mdd > 0.0:
        # Active had zero drawdown; any non-zero candidate drawdown is a
        # material increase relative to the baseline.
        return _result(False, REASON_MAX_DRAWDOWN_INCREASED, details)

    # 6) stability floor
    if (
        float(details["candidate_stability"]) < config.min_stability_score
    ):
        return _result(False, REASON_STABILITY_TOO_LOW, details)

    # All checks passed.
    return _result(True, REASON_OK, details)


# ---- window split helper -------------------------------------------------
def _month_key_of(record):
    date = record.get("date")
    if not isinstance(date, str) or len(date) < 7:
        raise ValueError(
            "trade record 'date' must be a string starting with YYYY-MM"
        )
    return date[:7]


def split_trades_into_oos_windows(trades, validation_months=1):
    """Split a list of completed trades into discovery + validation lists.

    Trades are bucketed by month-key (YYYY-MM). The latest `validation_months`
    months go to the validation list; everything earlier goes to discovery.

    Returns:
      {
        "discovery_trades": [...],
        "validation_trades": [...],
        "discovery_window": (start_month, end_month) or None,
        "validation_window": (start_month, end_month) or None,
      }
    """
    if not isinstance(trades, list):
        raise ValueError("trades must be a list")
    if not isinstance(validation_months, int) or validation_months <= 0:
        raise ValueError("validation_months must be a positive int")

    months = set()
    for t in trades:
        months.add(_month_key_of(t))
    sorted_months = sorted(months)
    if len(sorted_months) <= validation_months:
        # Not enough months for a meaningful split.
        return {
            "discovery_trades": list(trades),
            "validation_trades": [],
            "discovery_window": (
                (sorted_months[0], sorted_months[-1]) if sorted_months else None
            ),
            "validation_window": None,
        }

    validation_month_set = set(sorted_months[-validation_months:])
    discovery_month_set = set(sorted_months[:-validation_months])

    discovery_trades = []
    validation_trades = []
    for t in trades:
        m = _month_key_of(t)
        if m in validation_month_set:
            validation_trades.append(t)
        else:
            discovery_trades.append(t)

    discovery_sorted = sorted(discovery_month_set)
    validation_sorted = sorted(validation_month_set)
    return {
        "discovery_trades": discovery_trades,
        "validation_trades": validation_trades,
        "discovery_window": (discovery_sorted[0], discovery_sorted[-1]),
        "validation_window": (validation_sorted[0], validation_sorted[-1]),
    }


__all__ = [
    "OOSValidationConfig",
    "REASON_INSUFFICIENT_VALIDATION_TRADES",
    "REASON_MAX_DRAWDOWN_INCREASED",
    "REASON_OK",
    "REASON_PROFIT_FACTOR_DETERIORATED",
    "REASON_STABILITY_TOO_LOW",
    "REASON_TRADE_COUNT_REDUCED",
    "REASON_VALIDATION_NOT_AFTER_DISCOVERY",
    "REASON_WIN_RATE_BUT_RISK_WORSENS",
    "evaluate_oos_promotion",
    "split_trades_into_oos_windows",
]
