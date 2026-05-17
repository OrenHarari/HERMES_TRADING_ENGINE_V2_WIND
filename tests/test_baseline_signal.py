"""Tests for hermes.signals.baseline_signal (Phase 3A)."""

import unittest

from hermes.signals.baseline_signal import (
    BASELINE_LOOKBACK,
    NEUTRAL_TRIPLE,
    baseline_signal,
)


def _flat(n=20, p=100.0):
    return [{"timestamp": i, "open": p, "high": p, "low": p,
             "close": p, "volume": 1.0} for i in range(n)]


def _uptrend(n=20, start=100.0, step=1.0):
    return [{"timestamp": i, "open": start + i * step,
             "high": start + i * step, "low": start + i * step,
             "close": start + i * step, "volume": 1.0} for i in range(n)]


def _downtrend(n=20, start=100.0, step=1.0):
    return [{"timestamp": i, "open": start - i * step,
             "high": start - i * step, "low": start - i * step,
             "close": start - i * step, "volume": 1.0} for i in range(n)]


class TestBaselineSignal(unittest.TestCase):
    def test_returns_three_keys(self):
        out = baseline_signal(_uptrend(20), 19)
        self.assertEqual(
            set(out.keys()),
            {"sequence_value", "amd_value", "combined_value"},
        )

    def test_values_in_unit_interval(self):
        for c in (_flat(20), _uptrend(20), _downtrend(20)):
            for i in (0, 5, 9, 10, 19):
                out = baseline_signal(c, i)
                for k, v in out.items():
                    self.assertGreaterEqual(v, 0.0, msg=k)
                    self.assertLessEqual(v, 1.0, msg=k)

    def test_neutral_on_cold_start(self):
        c = _uptrend(20)
        for i in range(BASELINE_LOOKBACK - 1):
            self.assertEqual(baseline_signal(c, i), NEUTRAL_TRIPLE)

    def test_deterministic(self):
        c = _uptrend(20)
        a = baseline_signal(c, 15)
        b = baseline_signal(c, 15)
        self.assertEqual(a, b)

    def test_no_future_data(self):
        c = _uptrend(20)
        before = baseline_signal(c, 12)
        # Mutate candles[13:] arbitrarily.
        for j in range(13, 20):
            c[j]["close"] = 1e9
        after = baseline_signal(c, 12)
        self.assertEqual(before, after)

    def test_uptrend_pushes_sequence_above_neutral(self):
        out = baseline_signal(_uptrend(20, step=1.0), 19)
        self.assertGreater(out["sequence_value"], 0.5)

    def test_downtrend_pushes_sequence_below_neutral(self):
        out = baseline_signal(_downtrend(20, step=1.0), 19)
        self.assertLess(out["sequence_value"], 0.5)

    def test_flat_input_is_neutral_after_cold_start(self):
        out = baseline_signal(_flat(20), 19)
        # trend = 0 -> sequence = 0.5, amd = 0.0, combined = 0.25
        self.assertAlmostEqual(out["sequence_value"], 0.5, places=9)
        self.assertAlmostEqual(out["amd_value"], 0.0, places=9)
        self.assertAlmostEqual(out["combined_value"], 0.25, places=9)

    def test_known_value_uptrend(self):
        # Constructed: close[i-9] = 100, close[i] = 110 -> trend = 0.10
        # sequence = clamp01(0.5 + 0.5) = 1.0
        # amd      = clamp01(min(1.0, 0.10 * 50)) = clamp01(5.0) = 1.0
        # combined = (1.0 + 1.0) / 2 = 1.0
        c = []
        for i in range(10):
            p = 100.0 + i * (10.0 / 9.0)
            c.append({"timestamp": i, "open": p, "high": p, "low": p,
                      "close": p, "volume": 1.0})
        out = baseline_signal(c, 9)
        self.assertAlmostEqual(out["sequence_value"], 1.0, places=9)
        self.assertAlmostEqual(out["amd_value"], 1.0, places=9)
        self.assertAlmostEqual(out["combined_value"], 1.0, places=9)

    def test_invalid_index_rejected(self):
        c = _flat(10)
        with self.assertRaises(ValueError):
            baseline_signal(c, -1)
        with self.assertRaises(ValueError):
            baseline_signal(c, 10)


if __name__ == "__main__":
    unittest.main()
