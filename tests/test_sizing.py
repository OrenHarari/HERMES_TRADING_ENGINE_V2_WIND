"""Tests for Prompt 1 / Step 5 Part 3 - Baseline position sizing."""

import unittest

from hermes.risk.sizing import compute_position_size


class TestComputePositionSize(unittest.TestCase):
    def test_zero_confidence_zero_size(self):
        self.assertEqual(compute_position_size(100.0, 0.0), 0.0)

    def test_full_confidence_full_size(self):
        self.assertEqual(compute_position_size(100.0, 1.0), 100.0)

    def test_scales_linearly_with_confidence(self):
        self.assertEqual(compute_position_size(100.0, 0.5), 50.0)
        self.assertEqual(compute_position_size(100.0, 0.25), 25.0)

    def test_returns_float(self):
        self.assertIsInstance(compute_position_size(100, 0.5), float)

    def test_deterministic(self):
        a = compute_position_size(100.0, 0.7)
        b = compute_position_size(100.0, 0.7)
        self.assertEqual(a, b)

    def test_cap_below_confidence_caps_size(self):
        # confidence=1.0 but cap=0.5 -> result is 0.5 * base.
        self.assertEqual(
            compute_position_size(100.0, 1.0, confidence_multiplier_cap=0.5),
            50.0,
        )

    def test_cap_above_confidence_does_not_increase_size(self):
        # confidence=0.4, cap=2.0 -> result is 0.4 * base (cap doesn't lift).
        self.assertEqual(
            compute_position_size(100.0, 0.4, confidence_multiplier_cap=2.0),
            40.0,
        )

    def test_zero_base_size_returns_zero(self):
        self.assertEqual(compute_position_size(0.0, 1.0), 0.0)

    def test_rejects_negative_base(self):
        with self.assertRaises(ValueError):
            compute_position_size(-1.0, 0.5)

    def test_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            compute_position_size(100.0, 1.5)
        with self.assertRaises(ValueError):
            compute_position_size(100.0, -0.01)
        with self.assertRaises(ValueError):
            compute_position_size(100.0, "0.5")
        with self.assertRaises(ValueError):
            compute_position_size(100.0, True)

    def test_rejects_zero_or_negative_cap(self):
        with self.assertRaises(ValueError):
            compute_position_size(100.0, 0.5, confidence_multiplier_cap=0.0)
        with self.assertRaises(ValueError):
            compute_position_size(100.0, 0.5, confidence_multiplier_cap=-1.0)


if __name__ == "__main__":
    unittest.main()
