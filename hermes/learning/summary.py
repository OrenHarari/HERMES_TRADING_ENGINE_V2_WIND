"""Step 6 - Canonical learning-output summary.

Produces the dict required by the Prompt 1 stop condition:

  {
    "total_trades": int,
    "overall_win_rate": float,
    "best_conditions": [...],
    "worst_conditions": [...],
    "thresholds_adapted": bool,
    "edge_decay_alert": bool,
  }

Keys are exact (Prompt 1). Phase 2 will add additive optional keys without
removing or renaming any of these.
"""

REQUIRED_LEARNING_SUMMARY_KEYS = (
    "total_trades",
    "overall_win_rate",
    "best_conditions",
    "worst_conditions",
    "thresholds_adapted",
    "edge_decay_alert",
)


def _overall_win_rate(trades):
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0.0)
    return float(wins) / float(len(trades))


def build_learning_summary(
    trades, attribution_result, threshold_result, edge_decay_state
):
    """Assemble the canonical learning summary dict.

    `trades`: list of completed-trade dicts (or anything with .__iter__).
    `attribution_result`: dict from compute_attribution.
    `threshold_result`: dict from ThresholdAdapter.propose.
    `edge_decay_state`: dict from EdgeDecayMonitor.state.

    Output dict has EXACTLY the keys in REQUIRED_LEARNING_SUMMARY_KEYS.
    """
    trades_list = list(trades)
    if not isinstance(attribution_result, dict):
        raise ValueError("attribution_result must be a dict")
    if not isinstance(threshold_result, dict):
        raise ValueError("threshold_result must be a dict")
    if not isinstance(edge_decay_state, dict):
        raise ValueError("edge_decay_state must be a dict")
    return {
        "total_trades": len(trades_list),
        "overall_win_rate": _overall_win_rate(trades_list),
        "best_conditions": list(attribution_result.get("best_conditions", [])),
        "worst_conditions": list(attribution_result.get("worst_conditions", [])),
        "thresholds_adapted": bool(threshold_result.get("thresholds_adapted", False)),
        "edge_decay_alert": bool(edge_decay_state.get("edge_decay_alert", False)),
    }
