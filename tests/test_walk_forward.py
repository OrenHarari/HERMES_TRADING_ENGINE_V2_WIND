"""Tests for Prompt 1 / Step 6 Part 4 - Walk-Forward."""

import unittest

from hermes.learning.walk_forward import walk_forward_analysis


def _trade(date, pnl, regime="trend_up"):
    return {"date": date, "regime": regime, "pnl": pnl, "net_pnl": pnl}


def _month_records(year_month, n_wins=8, n_losses=2):
    out = []
    for i in range(n_wins):
        out.append(_trade("{!s}-{:02d}".format(year_month, (i % 28) + 1), 10.0))
    for i in range(n_losses):
        out.append(_trade("{!s}-{:02d}".format(year_month, (i % 28) + 1), -5.0))
    return out


class TestInsufficientData(unittest.TestCase):
    def test_returns_insufficient_when_too_few_months(self):
        trades = _month_records("2025-01") + _month_records("2025-02")
        out = walk_forward_analysis(trades, train_months=6, test_months=1)
        self.assertEqual(out["windows"], [])
        self.assertTrue(out["summary"]["insufficient_data"])
        self.assertFalse(out["summary"]["is_consistent"])


class TestWindowingNonOverlap(unittest.TestCase):
    def test_train_strictly_before_test(self):
        months = ["2025-{:02d}".format(m) for m in range(1, 12)]
        trades = []
        for m in months:
            trades.extend(_month_records(m))
        out = walk_forward_analysis(trades, train_months=6, test_months=1)
        for w in out["windows"]:
            self.assertLess(w["train_end"], w["test_start"])
            # No overlap: ensure month sets are disjoint.
            train_set = {m for m in months if w["train_start"] <= m <= w["train_end"]}
            test_set = {m for m in months if w["test_start"] <= m <= w["test_end"]}
            self.assertTrue(train_set.isdisjoint(test_set))

    def test_window_count(self):
        months = ["2025-{:02d}".format(m) for m in range(1, 13)]  # 12 months
        trades = []
        for m in months:
            trades.extend(_month_records(m))
        out = walk_forward_analysis(trades, train_months=6, test_months=1)
        # 12 - 6 - 1 + 1 = 6 windows
        self.assertEqual(len(out["windows"]), 6)


class TestNoFutureDataInTraining(unittest.TestCase):
    def test_replacing_future_months_does_not_change_earlier_windows(self):
        months = ["2025-{:02d}".format(m) for m in range(1, 13)]
        baseline = []
        for m in months:
            baseline.extend(_month_records(m))
        out_all = walk_forward_analysis(baseline, train_months=6, test_months=1)
        # Drop the LAST month entirely and re-run; the first window's report
        # must be unchanged because its train+test are entirely earlier.
        truncated = [t for t in baseline if t["date"][:7] != "2025-12"]
        out_trunc = walk_forward_analysis(truncated, train_months=6, test_months=1)
        self.assertEqual(out_all["windows"][0], out_trunc["windows"][0])


class TestEdgeDecayFlag(unittest.TestCase):
    def test_flag_raised_on_winrate_drop_over_15pct(self):
        # Window 1: 90% win, Window 2: 60% win -> drop = 30% > 15%
        trades = []
        # Months 1-6 = train of window 1 (irrelevant to test metrics)
        # Month 7 = test of window 1: 9 wins / 1 loss = 90%
        for m in range(1, 7):
            trades.extend(_month_records("2025-{:02d}".format(m)))
        trades.extend(_month_records("2025-07", n_wins=9, n_losses=1))
        # Month 8 = test of window 2: 6 wins / 4 losses = 60%
        trades.extend(_month_records("2025-08", n_wins=6, n_losses=4))
        out = walk_forward_analysis(trades, train_months=6, test_months=1)
        self.assertTrue(out["summary"]["edge_decay_flag"])

    def test_flag_not_raised_on_stable_winrate(self):
        trades = []
        for m in range(1, 9):
            trades.extend(_month_records("2025-{:02d}".format(m), n_wins=7, n_losses=3))
        out = walk_forward_analysis(trades, train_months=6, test_months=1)
        self.assertFalse(out["summary"]["edge_decay_flag"])


class TestIsConsistent(unittest.TestCase):
    def test_inconsistent_when_some_window_rejected(self):
        # Build a series where one window has a very low win_rate.
        trades = []
        for m in range(1, 7):
            trades.extend(_month_records("2025-{:02d}".format(m)))
        trades.extend(_month_records("2025-07", n_wins=2, n_losses=8))  # 20% wr
        trades.extend(_month_records("2025-08", n_wins=8, n_losses=2))  # 80% wr
        out = walk_forward_analysis(trades, train_months=6, test_months=1)
        self.assertFalse(out["summary"]["is_consistent"])


class TestRejectsBadInputs(unittest.TestCase):
    def test_rejects_non_list(self):
        with self.assertRaises(ValueError):
            walk_forward_analysis("not a list")

    def test_rejects_zero_months(self):
        with self.assertRaises(ValueError):
            walk_forward_analysis([], train_months=0)

    def test_rejects_record_without_date(self):
        with self.assertRaises(ValueError):
            walk_forward_analysis([{"pnl": 1.0}])


if __name__ == "__main__":
    unittest.main()
