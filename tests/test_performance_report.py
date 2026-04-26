"""Tests for Prompt 1 / Step 5 Part 5 - Performance Report."""

import math
import unittest

from hermes.decision.performance import (
    PERFORMANCE_REPORT_KEYS,
    compute_performance_report,
)


class TestPerformanceReport(unittest.TestCase):
    def test_required_keys_match_spec(self):
        self.assertEqual(
            set(PERFORMANCE_REPORT_KEYS),
            {
                "net_pnl", "win_rate", "avg_win", "avg_loss",
                "profit_factor", "max_drawdown", "trade_count",
                "trades_per_regime", "stability_score", "cost_model_applied",
            },
        )

    def test_empty_trades_returns_zeroed_report(self):
        rep = compute_performance_report([])
        self.assertEqual(set(rep.keys()), set(PERFORMANCE_REPORT_KEYS))
        self.assertEqual(rep["net_pnl"], 0.0)
        self.assertEqual(rep["win_rate"], 0.0)
        self.assertEqual(rep["trade_count"], 0)
        self.assertEqual(rep["trades_per_regime"], {})
        self.assertEqual(rep["stability_score"], 0.0)
        self.assertEqual(rep["cost_model_applied"], False)

    def test_cost_model_applied_is_false_in_phase1(self):
        # Per spec: Prompt 1 must mark cost_model_applied = false.
        rep = compute_performance_report([{"net_pnl": 10.0, "regime": "trend_up"}])
        self.assertEqual(rep["cost_model_applied"], False)

    def test_basic_metrics(self):
        trades = [
            {"net_pnl": 10.0, "regime": "trend_up"},
            {"net_pnl": -5.0, "regime": "chop"},
            {"net_pnl": 20.0, "regime": "trend_up"},
            {"net_pnl": -3.0, "regime": "trend_up"},
        ]
        rep = compute_performance_report(trades)
        self.assertEqual(rep["trade_count"], 4)
        self.assertEqual(rep["net_pnl"], 22.0)
        self.assertAlmostEqual(rep["win_rate"], 0.5, places=12)
        self.assertEqual(rep["avg_win"], 15.0)
        self.assertEqual(rep["avg_loss"], -4.0)
        self.assertAlmostEqual(rep["profit_factor"], 30.0 / 8.0, places=12)
        self.assertEqual(rep["trades_per_regime"], {"trend_up": 3, "chop": 1})

    def test_max_drawdown_computed_correctly(self):
        trades = [
            {"net_pnl": 100.0},
            {"net_pnl": -40.0},
            {"net_pnl": -30.0},   # equity now at 30, peak was 100 -> dd = 70
            {"net_pnl": 50.0},
            {"net_pnl": -20.0},
        ]
        rep = compute_performance_report(trades)
        self.assertEqual(rep["max_drawdown"], 70.0)

    def test_max_drawdown_zero_when_monotonic_up(self):
        trades = [{"net_pnl": x} for x in (5.0, 10.0, 3.0, 2.0)]
        rep = compute_performance_report(trades)
        self.assertEqual(rep["max_drawdown"], 0.0)

    def test_profit_factor_inf_when_no_losses(self):
        rep = compute_performance_report([{"net_pnl": 10.0}, {"net_pnl": 20.0}])
        self.assertTrue(math.isinf(rep["profit_factor"]))
        self.assertGreater(rep["profit_factor"], 0)

    def test_profit_factor_zero_when_no_wins_no_losses(self):
        rep = compute_performance_report([{"net_pnl": 0.0}])
        self.assertEqual(rep["profit_factor"], 0.0)

    def test_pnl_fallback_when_net_pnl_absent(self):
        rep = compute_performance_report([{"pnl": 5.0}, {"pnl": -2.0}])
        self.assertEqual(rep["net_pnl"], 3.0)

    def test_rejects_record_without_pnl(self):
        with self.assertRaises(ValueError):
            compute_performance_report([{"regime": "x"}])

    def test_stability_score_in_unit_interval(self):
        for trades in [
            [{"net_pnl": 1.0}],
            [{"net_pnl": -1.0}],
            [{"net_pnl": 10.0}, {"net_pnl": -50.0}, {"net_pnl": 100.0}],
        ]:
            rep = compute_performance_report(trades)
            self.assertGreaterEqual(rep["stability_score"], 0.0)
            self.assertLessEqual(rep["stability_score"], 1.0)

    def test_deterministic(self):
        trades = [{"net_pnl": 1.0}, {"net_pnl": -2.0}, {"net_pnl": 3.0}]
        a = compute_performance_report(trades)
        b = compute_performance_report(trades)
        self.assertEqual(a, b)

    def test_trades_per_regime_handles_missing_regime(self):
        rep = compute_performance_report([{"net_pnl": 1.0}, {"net_pnl": 2.0, "regime": "trend_up"}])
        self.assertEqual(rep["trades_per_regime"].get("", 0), 1)
        self.assertEqual(rep["trades_per_regime"].get("trend_up", 0), 1)

    def test_rejects_non_list(self):
        with self.assertRaises(ValueError):
            compute_performance_report("not a list")


if __name__ == "__main__":
    unittest.main()
