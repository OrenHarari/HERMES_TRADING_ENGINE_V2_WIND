"""Step 6 / Part 2 - Signal Attribution Engine.

Computes win-rate breakdowns over completed trades along three axes:
  1. confidence buckets
  2. regimes
  3. confidence_bucket x regime cross-analysis

Cells with sample size below the configured minimum are EXCLUDED entirely
(per spec: "Do not infer from tiny samples"). They are not reported as
"low confidence" results, they simply do not appear.

Top-N best/worst conditions are sorted by win_rate (desc/asc), with
deterministic tie-breakers on trade_count then condition string.
"""

from hermes.decision.performance import profit_factor_from_sums
from hermes.market import REGIME_VALUES

# Fixed 10 confidence buckets [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
# (the last bucket is closed on the right to include the value 1.0).
BUCKET_BOUNDARIES = (
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
)


def bucket_label_for_confidence(confidence):
    """Return the bucket label string for a confidence value in [0, 1]."""
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be in [0, 1]; got {!r}".format(confidence))
    for i, (lo, hi) in enumerate(BUCKET_BOUNDARIES):
        is_last = i == len(BUCKET_BOUNDARIES) - 1
        if (confidence >= lo) and (confidence < hi or (is_last and confidence == 1.0)):
            return "confidence={lo:.1f}-{hi:.1f}".format(lo=lo, hi=hi)
    raise ValueError("could not bucket confidence: {!r}".format(confidence))


def bucket_lower_bound_for_label(label):
    """Inverse: extract the lower bound from a bucket label string."""
    # Format: 'confidence=0.6-0.7'
    body = label.split("=", 1)[1]
    lo_str = body.split("-", 1)[0]
    return float(lo_str)


class AttributionConfig(object):
    """Configurable thresholds for attribution.

    Defaults match Prompt 1 Part 2 baseline. Tests may inject smaller values.
    """

    __slots__ = (
        "min_trades_per_bucket",
        "min_trades_per_regime",
        "min_trades_per_combination",
        "top_n_best",
        "top_n_worst",
    )

    def __init__(
        self,
        min_trades_per_bucket=20,
        min_trades_per_regime=30,
        min_trades_per_combination=50,
        top_n_best=5,
        top_n_worst=5,
    ):
        for name, val in (
            ("min_trades_per_bucket", min_trades_per_bucket),
            ("min_trades_per_regime", min_trades_per_regime),
            ("min_trades_per_combination", min_trades_per_combination),
            ("top_n_best", top_n_best),
            ("top_n_worst", top_n_worst),
        ):
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ValueError("{!s} must be int >= 0".format(name))
        self.min_trades_per_bucket = min_trades_per_bucket
        self.min_trades_per_regime = min_trades_per_regime
        self.min_trades_per_combination = min_trades_per_combination
        self.top_n_best = top_n_best
        self.top_n_worst = top_n_worst


def _aggregate(records):
    """Compute (trade_count, win_rate, avg_net_pnl, profit_factor) for a list
    of completed-trade dicts.

    profit_factor: sum_wins / abs(sum_losses); if no losses -> float('inf')
    if there were wins, otherwise 0.0.
    """
    n = len(records)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    wins = [r for r in records if r["pnl"] > 0.0]
    losses = [r for r in records if r["pnl"] < 0.0]
    win_rate = float(len(wins)) / float(n)

    def _net(r):
        return float(r.get("net_pnl", r["pnl"]))

    avg_net_pnl = sum(_net(r) for r in records) / float(n)
    sum_wins = sum(_net(r) for r in wins) if wins else 0.0
    sum_losses_abs = -sum(_net(r) for r in losses) if losses else 0.0
    profit_factor = profit_factor_from_sums(sum_wins, sum_losses_abs)
    return n, win_rate, avg_net_pnl, float(profit_factor)


def _build_entry(condition, records):
    n, win_rate, avg_net_pnl, profit_factor = _aggregate(records)
    return {
        "condition": condition,
        "trade_count": n,
        "win_rate": win_rate,
        "avg_net_pnl": avg_net_pnl,
        "profit_factor": profit_factor,
    }


def compute_attribution(trades, config=None):
    """Compute attribution analysis over a list of completed trade dicts.

    Returns:
      {
        "by_bucket":      [entry, ...],   # only buckets with enough samples
        "by_regime":      [entry, ...],
        "by_combination": [entry, ...],
        "best_conditions":  [entry, ...], # top-N across all three axes
        "worst_conditions": [entry, ...], # bottom-N across all three axes
      }

    Tie-breakers (deterministic): higher trade_count first, then condition
    string ascending.
    """
    if config is None:
        config = AttributionConfig()
    if not isinstance(trades, list):
        raise ValueError("trades must be a list")

    # Bucket grouping.
    by_bucket_records = {label_lo[0]: [] for label_lo in [
        ("confidence={:.1f}-{:.1f}".format(lo, hi),) for (lo, hi) in BUCKET_BOUNDARIES
    ]}
    by_regime_records = {r: [] for r in REGIME_VALUES}
    by_combination_records = {}

    for t in trades:
        if "confidence" not in t or "regime" not in t or "pnl" not in t:
            raise ValueError(
                "trade record missing one of: confidence, regime, pnl"
            )
        bucket = bucket_label_for_confidence(t["confidence"])
        by_bucket_records[bucket].append(t)
        regime = t["regime"]
        if regime in by_regime_records:
            by_regime_records[regime].append(t)
        combo = "{!s}|regime={!s}".format(bucket, regime)
        by_combination_records.setdefault(combo, []).append(t)

    by_bucket = []
    for bucket_label in [
        "confidence={:.1f}-{:.1f}".format(lo, hi) for (lo, hi) in BUCKET_BOUNDARIES
    ]:
        recs = by_bucket_records[bucket_label]
        if len(recs) >= config.min_trades_per_bucket:
            by_bucket.append(_build_entry(bucket_label, recs))

    by_regime = []
    for regime_label in REGIME_VALUES:
        recs = by_regime_records[regime_label]
        if len(recs) >= config.min_trades_per_regime:
            by_regime.append(
                _build_entry("regime={!s}".format(regime_label), recs)
            )

    by_combination = []
    for combo_label in sorted(by_combination_records.keys()):
        recs = by_combination_records[combo_label]
        if len(recs) >= config.min_trades_per_combination:
            by_combination.append(_build_entry(combo_label, recs))

    all_entries = list(by_bucket) + list(by_regime) + list(by_combination)

    def _key_desc(entry):
        # Sort by win_rate desc, trade_count desc, condition asc
        return (-entry["win_rate"], -entry["trade_count"], entry["condition"])

    def _key_asc(entry):
        return (entry["win_rate"], -entry["trade_count"], entry["condition"])

    best = sorted(all_entries, key=_key_desc)[: config.top_n_best]
    worst = sorted(all_entries, key=_key_asc)[: config.top_n_worst]

    return {
        "by_bucket": by_bucket,
        "by_regime": by_regime,
        "by_combination": by_combination,
        "best_conditions": best,
        "worst_conditions": worst,
    }
