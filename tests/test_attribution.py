"""Tests for Prompt 1 / Step 6 Part 2 - Signal Attribution."""

import unittest

from hermes.learning import (
    AttributionConfig,
    BUCKET_BOUNDARIES,
    bucket_label_for_confidence,
    compute_attribution,
)
from hermes.market import REGIME_CHOP, REGIME_TREND_UP


def _trade(confidence, regime, pnl, net_pnl=None):
    t = {"confidence": confidence, "regime": regime, "pnl": pnl}
    if net_pnl is not None:
        t["net_pnl"] = net_pnl
    return t


class TestBucketLabel(unittest.TestCase):
    def test_bucket_count(self):
        self.assertEqual(len(BUCKET_BOUNDARIES), 10)

    def test_bucket_labels_for_known_values(self):
        self.assertEqual(bucket_label_for_confidence(0.0), "confidence=0.0-0.1")
        self.assertEqual(bucket_label_for_confidence(0.05), "confidence=0.0-0.1")
        self.assertEqual(bucket_label_for_confidence(0.1), "confidence=0.1-0.2")
        self.assertEqual(bucket_label_for_confidence(0.65), "confidence=0.6-0.7")
        self.assertEqual(bucket_label_for_confidence(1.0), "confidence=0.9-1.0")

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            bucket_label_for_confidence(-0.01)
        with self.assertRaises(ValueError):
            bucket_label_for_confidence(1.01)


class TestComputeAttributionMinSamples(unittest.TestCase):
    def test_returns_empty_lists_for_no_trades(self):
        out = compute_attribution([])
        self.assertEqual(out["by_bucket"], [])
        self.assertEqual(out["by_regime"], [])
        self.assertEqual(out["by_combination"], [])
        self.assertEqual(out["best_conditions"], [])
        self.assertEqual(out["worst_conditions"], [])

    def test_excludes_buckets_below_min(self):
        # 3 trades in same bucket / regime - far below default minimums.
        trades = [_trade(0.65, REGIME_TREND_UP, p) for p in (1.0, 2.0, 3.0)]
        out = compute_attribution(trades)
        self.assertEqual(out["by_bucket"], [])
        self.assertEqual(out["by_regime"], [])
        self.assertEqual(out["by_combination"], [])

    def test_includes_buckets_when_min_met(self):
        cfg = AttributionConfig(
            min_trades_per_bucket=3,
            min_trades_per_regime=3,
            min_trades_per_combination=3,
        )
        trades = [_trade(0.65, REGIME_TREND_UP, 1.0) for _ in range(3)]
        out = compute_attribution(trades, cfg)
        self.assertEqual(len(out["by_bucket"]), 1)
        self.assertEqual(len(out["by_regime"]), 1)
        self.assertEqual(len(out["by_combination"]), 1)


class TestComputeAttributionMath(unittest.TestCase):
    def test_win_rate_and_pnl_metrics(self):
        cfg = AttributionConfig(
            min_trades_per_bucket=4, min_trades_per_regime=4,
            min_trades_per_combination=4,
        )
        # 3 wins, 1 loss in confidence 0.6-0.7 / trend_up.
        trades = [
            _trade(0.65, REGIME_TREND_UP, 10.0, net_pnl=10.0),
            _trade(0.65, REGIME_TREND_UP, 20.0, net_pnl=20.0),
            _trade(0.65, REGIME_TREND_UP, -5.0, net_pnl=-5.0),
            _trade(0.65, REGIME_TREND_UP, 15.0, net_pnl=15.0),
        ]
        out = compute_attribution(trades, cfg)
        bucket = out["by_bucket"][0]
        self.assertEqual(bucket["trade_count"], 4)
        self.assertAlmostEqual(bucket["win_rate"], 0.75, places=12)
        self.assertAlmostEqual(bucket["avg_net_pnl"], 10.0, places=12)
        self.assertAlmostEqual(bucket["profit_factor"], 45.0 / 5.0, places=12)


class TestBestWorstConditions(unittest.TestCase):
    def test_best_sorted_by_winrate_desc(self):
        cfg = AttributionConfig(
            min_trades_per_bucket=2,
            min_trades_per_regime=2,
            min_trades_per_combination=2,
            top_n_best=10,
            top_n_worst=10,
        )
        # Bucket 0.6-0.7 / trend_up: 100% win
        # Bucket 0.7-0.8 / trend_up: 50% win
        trades = (
            [_trade(0.65, REGIME_TREND_UP, 1.0) for _ in range(2)]
            + [_trade(0.75, REGIME_TREND_UP, 1.0) for _ in range(1)]
            + [_trade(0.75, REGIME_TREND_UP, -1.0) for _ in range(1)]
        )
        out = compute_attribution(trades, cfg)
        wrs = [e["win_rate"] for e in out["best_conditions"]]
        self.assertEqual(wrs, sorted(wrs, reverse=True))

    def test_worst_sorted_by_winrate_asc(self):
        cfg = AttributionConfig(
            min_trades_per_bucket=2,
            min_trades_per_regime=2,
            min_trades_per_combination=2,
        )
        trades = (
            [_trade(0.65, REGIME_TREND_UP, 1.0) for _ in range(2)]
            + [_trade(0.45, REGIME_CHOP, -1.0) for _ in range(2)]
        )
        out = compute_attribution(trades, cfg)
        wrs = [e["win_rate"] for e in out["worst_conditions"]]
        self.assertEqual(wrs, sorted(wrs))


class TestAttributionConfig(unittest.TestCase):
    def test_default_minimums_match_spec(self):
        c = AttributionConfig()
        self.assertEqual(c.min_trades_per_bucket, 20)
        self.assertEqual(c.min_trades_per_regime, 30)
        self.assertEqual(c.min_trades_per_combination, 50)

    def test_negative_min_rejected(self):
        with self.assertRaises(ValueError):
            AttributionConfig(min_trades_per_bucket=-1)


class TestDeterministic(unittest.TestCase):
    def test_same_input_same_output(self):
        cfg = AttributionConfig(
            min_trades_per_bucket=2, min_trades_per_regime=2,
            min_trades_per_combination=2,
        )
        trades = [_trade(0.65, REGIME_TREND_UP, p) for p in (1.0, -1.0, 2.0)]
        a = compute_attribution(trades, cfg)
        b = compute_attribution(trades, cfg)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
