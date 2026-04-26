"""Tests for Prompt 1 / Step 6 Part 5 - Edge Decay."""

import unittest

from hermes.learning import EdgeDecayMonitor


def _feed(monitor, pnl_seq):
    state = None
    for p in pnl_seq:
        state = monitor.update_with_pnl(p)
    return state


class TestNoAlertEarly(unittest.TestCase):
    def test_no_alert_with_zero_trades(self):
        m = EdgeDecayMonitor()
        self.assertFalse(m.edge_decay_alert)

    def test_no_alert_after_one_window(self):
        m = EdgeDecayMonitor()
        # 20 wins -> one completed window with win_rate=1.0
        _feed(m, [1.0] * 20)
        self.assertFalse(m.edge_decay_alert)


class TestAlertTriggers(unittest.TestCase):
    def test_alerts_after_two_consecutive_low_windows(self):
        m = EdgeDecayMonitor()
        # Window 1: 8/20 = 40% < 0.45
        _feed(m, [1.0] * 8 + [-1.0] * 12)
        self.assertFalse(m.edge_decay_alert)
        # Window 2: 8/20 = 40% < 0.45 -> alert raised
        _feed(m, [1.0] * 8 + [-1.0] * 12)
        self.assertTrue(m.edge_decay_alert)
        self.assertEqual(m.proposed_min_confidence_bumps, 1)

    def test_does_not_alert_when_only_one_low_window(self):
        m = EdgeDecayMonitor()
        # Window 1 low.
        _feed(m, [1.0] * 8 + [-1.0] * 12)
        # Window 2 above threshold (12/20 = 60%).
        _feed(m, [1.0] * 12 + [-1.0] * 8)
        self.assertFalse(m.edge_decay_alert)

    def test_threshold_at_exactly_0_45_does_not_trigger(self):
        # Spec: rolling win_rate drops BELOW 0.45 (strictly less).
        m = EdgeDecayMonitor()
        # Two windows at exactly 9/20 = 0.45 -> NOT below.
        _feed(m, [1.0] * 9 + [-1.0] * 11)
        _feed(m, [1.0] * 9 + [-1.0] * 11)
        self.assertFalse(m.edge_decay_alert)


class TestRecovery(unittest.TestCase):
    def test_recovery_clears_alert_when_streak_above_0_50(self):
        m = EdgeDecayMonitor()
        # Trigger alert.
        _feed(m, [1.0] * 8 + [-1.0] * 12)
        _feed(m, [1.0] * 8 + [-1.0] * 12)
        self.assertTrue(m.edge_decay_alert)
        # Feed 10 trades with > 0.50 win rate (e.g. 6 wins / 4 losses = 0.6).
        _feed(m, [1.0] * 6 + [-1.0] * 4)
        self.assertFalse(m.edge_decay_alert)

    def test_recovery_does_not_clear_at_exactly_0_50(self):
        m = EdgeDecayMonitor()
        _feed(m, [1.0] * 8 + [-1.0] * 12)
        _feed(m, [1.0] * 8 + [-1.0] * 12)
        # 10 trades: 5/10 = 0.50, NOT > 0.50.
        _feed(m, [1.0] * 5 + [-1.0] * 5)
        self.assertTrue(m.edge_decay_alert)


class TestStateSnapshot(unittest.TestCase):
    def test_state_keys(self):
        m = EdgeDecayMonitor()
        s = m.state()
        for k in (
            "edge_decay_alert",
            "total_observations",
            "completed_windows",
            "proposed_min_confidence_bumps",
            "last_window_win_rates",
            "log",
        ):
            self.assertIn(k, s)


class TestDeterministic(unittest.TestCase):
    def test_same_pnl_sequence_same_state(self):
        seq = [1.0, -1.0, 1.0, -1.0] * 10
        a = EdgeDecayMonitor()
        b = EdgeDecayMonitor()
        for p in seq:
            a.update_with_pnl(p)
            b.update_with_pnl(p)
        self.assertEqual(a.state(), b.state())


class TestRejectsBadInputs(unittest.TestCase):
    def test_rejects_non_numeric(self):
        m = EdgeDecayMonitor()
        with self.assertRaises(ValueError):
            m.update_with_pnl("loss")
        with self.assertRaises(ValueError):
            m.update_with_pnl(True)
        with self.assertRaises(ValueError):
            m.update_with_pnl(float("nan"))


if __name__ == "__main__":
    unittest.main()
