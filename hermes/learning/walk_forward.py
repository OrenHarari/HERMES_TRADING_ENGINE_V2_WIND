"""Step 6 / Part 4 - Walk-Forward Parameter Update.

Rolling train/test analysis with strict no-overlap and no-future-data rules.

Trades are bucketed by the 'date' field (YYYY-MM-DD); month key = 'YYYY-MM'.

Per-window report contains:
  window_index, train_start, train_end, test_start, test_end, trade_count,
  win_rate, net_pnl, profit_factor, max_drawdown, stability_score,
  accepted, rejection_reason

Final aggregate report:
  avg_win_rate, avg_net_pnl, avg_profit_factor, worst_max_drawdown, is_consistent
"""

import math

from hermes.decision.performance import compute_performance_report

DEFAULT_TRAIN_MONTHS = 6
DEFAULT_TEST_MONTHS = 1
DEFAULT_MIN_WINDOW_TRADES = 20
DEFAULT_STABILITY_FLOOR = 0.30
DEFAULT_WINDOW_WIN_RATE_FLOOR = 0.45
DEFAULT_DRAWDOWN_RATIO_CEILING = 0.40
DEFAULT_PROFIT_FACTOR_FLOOR = 1.10


def _month_key_of(record):
    date = record.get("date")
    if not isinstance(date, str) or len(date) < 7:
        raise ValueError(
            "trade record 'date' must be a string starting with YYYY-MM"
        )
    return date[:7]


def _all_month_keys(trades):
    keys = set()
    for t in trades:
        keys.add(_month_key_of(t))
    return sorted(keys)


def _trades_in_months(trades, month_keys):
    keyset = set(month_keys)
    return [t for t in trades if _month_key_of(t) in keyset]


def _safe_avg(values):
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def walk_forward_analysis(
    trades,
    train_months=DEFAULT_TRAIN_MONTHS,
    test_months=DEFAULT_TEST_MONTHS,
    min_window_trades=DEFAULT_MIN_WINDOW_TRADES,
    stability_floor=DEFAULT_STABILITY_FLOOR,
    window_win_rate_floor=DEFAULT_WINDOW_WIN_RATE_FLOOR,
    drawdown_ratio_ceiling=DEFAULT_DRAWDOWN_RATIO_CEILING,
    profit_factor_floor=DEFAULT_PROFIT_FACTOR_FLOOR,
):
    """Run rolling-window walk-forward analysis on completed trades.

    Returns a dict:
      {
        "windows": [per-window report, ...],
        "summary": {avg_win_rate, avg_net_pnl, avg_profit_factor,
                    worst_max_drawdown, is_consistent, edge_decay_flag,
                    insufficient_data},
      }
    """
    if not isinstance(trades, list):
        raise ValueError("trades must be a list")
    if (
        not isinstance(train_months, int)
        or train_months < 1
        or not isinstance(test_months, int)
        or test_months < 1
    ):
        raise ValueError("train_months and test_months must be positive int")

    months = _all_month_keys(trades)
    if len(months) < train_months + test_months:
        return {
            "windows": [],
            "summary": {
                "avg_win_rate": 0.0,
                "avg_net_pnl": 0.0,
                "avg_profit_factor": 0.0,
                "worst_max_drawdown": 0.0,
                "is_consistent": False,
                "edge_decay_flag": False,
                "insufficient_data": True,
            },
        }

    windows = []
    prev_win_rate = None
    edge_decay_flag = False

    for i in range(0, len(months) - train_months - test_months + 1):
        train_keys = months[i : i + train_months]
        test_keys = months[i + train_months : i + train_months + test_months]
        # No overlap is guaranteed by slice math; assert defensively.
        assert set(train_keys).isdisjoint(set(test_keys)), "train/test overlap"
        # Future-data rule: train_keys must all be < test_keys[0].
        assert max(train_keys) < min(test_keys), "train must precede test"

        test_records = _trades_in_months(trades, test_keys)
        report = compute_performance_report(test_records)

        # Drawdown ratio normalized by absolute net_pnl when positive, else 1.
        denom = max(abs(report["net_pnl"]), 1.0)
        dd_ratio = report["max_drawdown"] / denom

        accepted = True
        rejection_reason = ""
        if report["trade_count"] < min_window_trades:
            accepted = False
            rejection_reason = "insufficient_window_trades"
        elif report["stability_score"] < stability_floor:
            accepted = False
            rejection_reason = "low_stability"
        elif report["win_rate"] < window_win_rate_floor:
            accepted = False
            rejection_reason = "low_win_rate"
        elif dd_ratio > drawdown_ratio_ceiling:
            accepted = False
            rejection_reason = "drawdown_ratio_high"
        elif (
            not math.isinf(report["profit_factor"])
            and report["profit_factor"] < profit_factor_floor
        ):
            accepted = False
            rejection_reason = "weak_profit_factor"

        # Edge decay flag: > 15% absolute drop vs. previous window.
        if (
            prev_win_rate is not None
            and (prev_win_rate - report["win_rate"]) > 0.15
        ):
            edge_decay_flag = True
        prev_win_rate = report["win_rate"]

        windows.append(
            {
                "window_index": len(windows),
                "train_start": train_keys[0],
                "train_end": train_keys[-1],
                "test_start": test_keys[0],
                "test_end": test_keys[-1],
                "trade_count": report["trade_count"],
                "win_rate": report["win_rate"],
                "net_pnl": report["net_pnl"],
                "profit_factor": report["profit_factor"],
                "max_drawdown": report["max_drawdown"],
                "stability_score": report["stability_score"],
                "accepted": accepted,
                "rejection_reason": rejection_reason,
            }
        )

    valid = [w for w in windows if w["accepted"]]
    win_rates = [w["win_rate"] for w in windows]
    pnls = [w["net_pnl"] for w in windows]
    pfs = [
        w["profit_factor"]
        for w in windows
        if not math.isinf(w["profit_factor"])
    ]
    drawdowns = [w["max_drawdown"] for w in windows]

    is_consistent = (
        len(windows) > 0
        and all(w["accepted"] for w in windows)
        and not edge_decay_flag
    )

    return {
        "windows": windows,
        "summary": {
            "avg_win_rate": _safe_avg(win_rates),
            "avg_net_pnl": _safe_avg(pnls),
            "avg_profit_factor": _safe_avg(pfs),
            "worst_max_drawdown": max(drawdowns) if drawdowns else 0.0,
            "is_consistent": is_consistent,
            "edge_decay_flag": edge_decay_flag,
            "insufficient_data": False,
            "valid_window_count": len(valid),
            "total_window_count": len(windows),
        },
    }
