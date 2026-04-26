"""Tests for Prompt 1 / Step 6 - Learning summary (canonical output)."""

import unittest

from hermes.learning import (
    AttributionConfig,
    EdgeDecayMonitor,
    REQUIRED_LEARNING_SUMMARY_KEYS,
    ThresholdAdapter,
    build_learning_summary,
    compute_attribution,
)
from hermes.market import REGIME_TREND_UP


def _trade(confidence, regime, pnl):
    return {"confidence": confidence, "regime": regime, "pnl": pnl}


class TestRequiredKeys(unittest.TestCase):
    def test_required_keys_match_spec(self):
        self.assertEqual(
            set(REQUIRED_LEARNING_SUMMARY_KEYS),
            {
                "total_trades",
                "overall_win_rate",
                "best_conditions",
                "worst_conditions",
                "thresholds_adapted",
                "edge_decay_alert",
            },
        )

    def test_summary_returns_exact_key_set(self):
        attrib = compute_attribution([])
        adapter = ThresholdAdapter()
        thr = adapter.propose(
            attrib,
            total_trades=0,
            active_thresholds={"min_confidence": 0.6, "allow_chop": False},
        )
        decay = EdgeDecayMonitor().state()
        out = build_learning_summary([], attrib, thr, decay)
        self.assertEqual(set(out.keys()), set(REQUIRED_LEARNING_SUMMARY_KEYS))


class TestSummaryValues(unittest.TestCase):
    def test_zero_trades_zero_winrate(self):
        attrib = compute_attribution([])
        adapter = ThresholdAdapter()
        thr = adapter.propose(
            attrib, total_trades=0,
            active_thresholds={"min_confidence": 0.6, "allow_chop": False},
        )
        decay = EdgeDecayMonitor().state()
        out = build_learning_summary([], attrib, thr, decay)
        self.assertEqual(out["total_trades"], 0)
        self.assertEqual(out["overall_win_rate"], 0.0)
        self.assertEqual(out["best_conditions"], [])
        self.assertEqual(out["worst_conditions"], [])
        self.assertFalse(out["thresholds_adapted"])
        self.assertFalse(out["edge_decay_alert"])

    def test_overall_win_rate_correct(self):
        trades = [
            _trade(0.65, REGIME_TREND_UP, 1.0),
            _trade(0.65, REGIME_TREND_UP, -1.0),
            _trade(0.65, REGIME_TREND_UP, 1.0),
            _trade(0.65, REGIME_TREND_UP, 1.0),
        ]
        cfg = AttributionConfig(
            min_trades_per_bucket=2, min_trades_per_regime=2,
            min_trades_per_combination=2,
        )
        attrib = compute_attribution(trades, cfg)
        adapter = ThresholdAdapter()
        thr = adapter.propose(
            attrib, total_trades=4,
            active_thresholds={"min_confidence": 0.6, "allow_chop": False},
        )
        decay = EdgeDecayMonitor().state()
        out = build_learning_summary(trades, attrib, thr, decay)
        self.assertEqual(out["total_trades"], 4)
        self.assertAlmostEqual(out["overall_win_rate"], 0.75, places=12)


class TestEdgeDecayFlagPropagated(unittest.TestCase):
    def test_alert_flag_appears_in_summary(self):
        m = EdgeDecayMonitor()
        # Two consecutive low windows -> alert raised.
        for _ in range(2):
            for _ in range(8):
                m.update_with_pnl(1.0)
            for _ in range(12):
                m.update_with_pnl(-1.0)
        attrib = compute_attribution([])
        adapter = ThresholdAdapter()
        thr = adapter.propose(
            attrib, total_trades=0,
            active_thresholds={"min_confidence": 0.6, "allow_chop": False},
        )
        out = build_learning_summary([], attrib, thr, m.state())
        self.assertTrue(out["edge_decay_alert"])


class TestThresholdsAdaptedFlag(unittest.TestCase):
    def test_thresholds_adapted_true_when_applied(self):
        # Build attribution with one bucket that meets target and is lower
        # than current min_confidence.
        attrib = {
            "by_bucket": [
                {
                    "condition": "confidence=0.5-0.6",
                    "win_rate": 0.65,
                    "trade_count": 100,
                    "avg_net_pnl": 0.0,
                    "profit_factor": 0.0,
                }
            ],
            "by_regime": [],
            "by_combination": [],
            "best_conditions": [],
            "worst_conditions": [],
        }
        adapter = ThresholdAdapter()
        thr = adapter.propose(
            attrib, total_trades=200,
            active_thresholds={"min_confidence": 0.70, "allow_chop": False},
        )
        out = build_learning_summary([], attrib, thr, EdgeDecayMonitor().state())
        self.assertTrue(out["thresholds_adapted"])

    def test_thresholds_adapted_false_in_addendum_mode(self):
        attrib = {
            "by_bucket": [
                {
                    "condition": "confidence=0.5-0.6",
                    "win_rate": 0.65,
                    "trade_count": 100,
                    "avg_net_pnl": 0.0,
                    "profit_factor": 0.0,
                }
            ],
            "by_regime": [],
            "by_combination": [],
        }
        adapter = ThresholdAdapter(safety_addendum_active=True)
        thr = adapter.propose(
            attrib, total_trades=200,
            active_thresholds={"min_confidence": 0.70, "allow_chop": False},
        )
        out = build_learning_summary([], attrib, thr, EdgeDecayMonitor().state())
        # Addendum-mode: thresholds_adapted stays False (Phase 2 contract).
        self.assertFalse(out["thresholds_adapted"])


class TestDeterministic(unittest.TestCase):
    def test_same_inputs_same_summary(self):
        attrib = compute_attribution([])
        adapter = ThresholdAdapter()
        thr = adapter.propose(
            attrib, total_trades=0,
            active_thresholds={"min_confidence": 0.6, "allow_chop": False},
        )
        decay = EdgeDecayMonitor().state()
        a = build_learning_summary([], attrib, thr, decay)
        b = build_learning_summary([], attrib, thr, decay)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
