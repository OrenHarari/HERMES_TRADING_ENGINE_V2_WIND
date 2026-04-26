"""Step 5 / Part 5 - Performance Report.

Computes a deterministic performance summary from a list of completed trade
records.

Output keys (exact set):
  - net_pnl
  - win_rate
  - avg_win
  - avg_loss
  - profit_factor       (float; may be 'inf' when no losses but wins > 0)
  - max_drawdown
  - trade_count
  - trades_per_regime   (dict[str -> int])
  - stability_score     (float in [0, 1])
  - cost_model_applied  (bool; PHASE 1 always False per Prompt 1 Part 5)

Phase 2 Step 5C will flip `cost_model_applied` to True when costs are
actually applied. The field is intentionally present in Phase 1 to be
honest about the absence of a cost model.

A "completed trade" record must contain at least:
  - net_pnl: numeric  (or pnl, used as fallback if net_pnl missing)
  - regime: str (optional; absent counts under the empty-string key "")
The function does not validate further; lifecycle/memory layers are
responsible for record integrity.
"""

from hermes.utils.bounds import clip01, is_numeric

PERFORMANCE_REPORT_KEYS = (
    "net_pnl",
    "win_rate",
    "avg_win",
    "avg_loss",
    "profit_factor",
    "max_drawdown",
    "trade_count",
    "trades_per_regime",
    "stability_score",
    "cost_model_applied",
)


def _coerce_pnl(record):
    if "net_pnl" in record:
        v = record["net_pnl"]
    elif "pnl" in record:
        v = record["pnl"]
    else:
        raise ValueError("trade record missing both 'net_pnl' and 'pnl'")
    if not is_numeric(v):
        raise ValueError("trade pnl must be numeric (non-bool, non-NaN); got {!r}".format(v))
    return float(v)


def profit_factor_from_sums(sum_wins, sum_losses_abs):
    """Return profit factor given sum of wins and absolute sum of losses.

    Conventions (shared with learning/attribution.py):
      - both zero -> 0.0
      - wins but no losses -> float('inf')
      - otherwise -> sum_wins / sum_losses_abs
    """
    if sum_wins == 0.0 and sum_losses_abs == 0.0:
        return 0.0
    if sum_losses_abs == 0.0:
        return float("inf")
    return sum_wins / sum_losses_abs


def compute_performance_report(completed_trades, cost_model_applied=False):
    """Compute the canonical performance report dict.

    `completed_trades` is a list of dicts. Order matters for drawdown.

    `cost_model_applied` (Prompt 2 / Step 5C, additive): when True, the report
    will mark `cost_model_applied=True`. Defaults to False to preserve the
    Prompt 1 honesty behavior when callers do not pass the flag.
    """
    if not isinstance(completed_trades, list):
        raise ValueError("completed_trades must be a list")
    if not isinstance(cost_model_applied, bool):
        raise ValueError("cost_model_applied must be a bool")

    trade_count = len(completed_trades)
    if trade_count == 0:
        return {
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "trades_per_regime": {},
            "stability_score": 0.0,
            "cost_model_applied": cost_model_applied,
        }

    pnls = [_coerce_pnl(r) for r in completed_trades]
    wins = [p for p in pnls if p > 0.0]
    losses = [p for p in pnls if p < 0.0]
    net_pnl = sum(pnls)
    win_count = len(wins)
    win_rate = float(win_count) / float(trade_count)
    avg_win = (sum(wins) / float(len(wins))) if wins else 0.0
    avg_loss = (sum(losses) / float(len(losses))) if losses else 0.0

    sum_wins = sum(wins) if wins else 0.0
    sum_losses_abs = sum(-p for p in losses) if losses else 0.0
    profit_factor = profit_factor_from_sums(sum_wins, sum_losses_abs)

    # Drawdown computed on cumulative equity curve.
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_drawdown:
            max_drawdown = dd

    trades_per_regime = {}
    for r in completed_trades:
        key = r.get("regime", "")
        if not isinstance(key, str):
            key = str(key)
        trades_per_regime[key] = trades_per_regime.get(key, 0) + 1

    # Stability score: deterministic, bounded blend of win_rate and the
    # proportion of peak equity preserved against drawdown.
    if peak > 0.0:
        normalized_dd = clip01(max_drawdown / peak)
    else:
        # No positive peak -> system never made money; stability is 0.
        normalized_dd = 1.0
    stability_score = clip01(win_rate * (1.0 - normalized_dd))

    return {
        "net_pnl": float(net_pnl),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "trade_count": int(trade_count),
        "trades_per_regime": trades_per_regime,
        "stability_score": float(stability_score),
        "cost_model_applied": cost_model_applied,
    }
