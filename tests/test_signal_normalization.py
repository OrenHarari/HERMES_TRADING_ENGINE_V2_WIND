"""Tests for Prompt 1 / Step 1 - Signal Normalization."""

import unittest

from hermes.signals import (
    REQUIRED_SIGNAL_KEYS,
    compute_agreement,
    normalize_signal,
    validate_signal,
)


class TestComputeAgreement(unittest.TestCase):
    def test_perfect_agreement(self):
        self.assertEqual(compute_agreement(0.5, 0.5), 1.0)
        self.assertEqual(compute_agreement(0.0, 0.0), 1.0)
        self.assertEqual(compute_agreement(1.0, 1.0), 1.0)

    def test_max_disagreement(self):
        self.assertEqual(compute_agreement(0.0, 1.0), 0.0)
        self.assertEqual(compute_agreement(1.0, 0.0), 0.0)

    def test_formula_correctness(self):
        # Spec: agreement = 1 - abs(sequence_value - amd_value)
        cases = [
            (0.7, 0.3, 0.6),
            (0.2, 0.5, 0.7),
            (0.9, 0.85, 0.95),
            (0.0, 0.25, 0.75),
            (1.0, 0.4, 0.4),
        ]
        for seq, amd, expected in cases:
            self.assertAlmostEqual(compute_agreement(seq, amd), expected, places=12)

    def test_bounded_in_unit_interval(self):
        for seq in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            for amd in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
                v = compute_agreement(seq, amd)
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            compute_agreement(-0.01, 0.5)
        with self.assertRaises(ValueError):
            compute_agreement(0.5, 1.01)
        with self.assertRaises(ValueError):
            compute_agreement(2.0, 0.5)

    def test_rejects_non_numeric(self):
        with self.assertRaises(ValueError):
            compute_agreement("0.5", 0.5)
        with self.assertRaises(ValueError):
            compute_agreement(0.5, None)
        with self.assertRaises(ValueError):
            compute_agreement([0.5], 0.5)

    def test_rejects_bool_no_implicit_cast(self):
        # bool is a subclass of int in Python; silent True->1.0 / False->0.0
        # would be implicit casting. Spec forbids it.
        with self.assertRaises(ValueError):
            compute_agreement(True, 0.5)
        with self.assertRaises(ValueError):
            compute_agreement(0.5, False)

    def test_rejects_nan_and_inf(self):
        nan = float("nan")
        inf = float("inf")
        with self.assertRaises(ValueError):
            compute_agreement(nan, 0.5)
        with self.assertRaises(ValueError):
            compute_agreement(0.5, nan)
        with self.assertRaises(ValueError):
            compute_agreement(inf, 0.5)
        with self.assertRaises(ValueError):
            compute_agreement(0.5, -inf)

    def test_deterministic(self):
        # Same input - same output, repeatedly.
        a = compute_agreement(0.31, 0.78)
        b = compute_agreement(0.31, 0.78)
        c = compute_agreement(0.31, 0.78)
        self.assertEqual(a, b)
        self.assertEqual(b, c)


class TestValidateSignal(unittest.TestCase):
    def test_required_keys_match_spec(self):
        # Sanity: spec lists exactly these three keys.
        self.assertEqual(
            set(REQUIRED_SIGNAL_KEYS),
            {"sequence_value", "amd_value", "combined_value"},
        )

    def test_accepts_valid_signal(self):
        self.assertIsNone(
            validate_signal(
                {"sequence_value": 0.5, "amd_value": 0.5, "combined_value": 0.5}
            )
        )

    def test_rejects_missing_keys(self):
        bases = [
            {"amd_value": 0.5, "combined_value": 0.5},
            {"sequence_value": 0.5, "combined_value": 0.5},
            {"sequence_value": 0.5, "amd_value": 0.5},
            {},
        ]
        for d in bases:
            with self.assertRaises(ValueError):
                validate_signal(d)

    def test_rejects_non_dict(self):
        for bad in (None, 0.5, "x", [], (0.5, 0.5, 0.5), 0):
            with self.assertRaises(ValueError):
                validate_signal(bad)

    def test_rejects_out_of_range_values(self):
        with self.assertRaises(ValueError):
            validate_signal(
                {"sequence_value": 1.0001, "amd_value": 0.5, "combined_value": 0.5}
            )
        with self.assertRaises(ValueError):
            validate_signal(
                {"sequence_value": -0.0001, "amd_value": 0.5, "combined_value": 0.5}
            )
        with self.assertRaises(ValueError):
            validate_signal(
                {"sequence_value": 0.5, "amd_value": 0.5, "combined_value": 2.0}
            )

    def test_rejects_nan_value(self):
        nan = float("nan")
        with self.assertRaises(ValueError):
            validate_signal(
                {"sequence_value": nan, "amd_value": 0.5, "combined_value": 0.5}
            )


class TestNormalizeSignal(unittest.TestCase):
    def test_returns_exact_required_key_set(self):
        out = normalize_signal(
            {"sequence_value": 0.6, "amd_value": 0.4, "combined_value": 0.5}
        )
        self.assertEqual(
            set(out.keys()),
            {"sequence_value", "amd_value", "combined_value", "agreement"},
        )

    def test_does_not_introduce_label(self):
        # Step 1 must NOT introduce labels (that is Step 2's responsibility).
        out = normalize_signal(
            {"sequence_value": 0.6, "amd_value": 0.4, "combined_value": 0.5}
        )
        self.assertNotIn("label", out)

    def test_agreement_in_output(self):
        out = normalize_signal(
            {"sequence_value": 0.6, "amd_value": 0.4, "combined_value": 0.5}
        )
        self.assertAlmostEqual(out["agreement"], 0.8, places=12)

    def test_values_in_unit_interval(self):
        for raw in [
            {"sequence_value": 0.0, "amd_value": 1.0, "combined_value": 0.5},
            {"sequence_value": 1.0, "amd_value": 0.0, "combined_value": 1.0},
            {"sequence_value": 0.123, "amd_value": 0.456, "combined_value": 0.789},
        ]:
            out = normalize_signal(raw)
            for k in ("sequence_value", "amd_value", "combined_value", "agreement"):
                self.assertGreaterEqual(out[k], 0.0)
                self.assertLessEqual(out[k], 1.0)

    def test_deterministic_same_input_same_output(self):
        raw = {"sequence_value": 0.31, "amd_value": 0.78, "combined_value": 0.55}
        a = normalize_signal(raw)
        b = normalize_signal(raw)
        self.assertEqual(a, b)
        # And again from a fresh equal dict.
        c = normalize_signal(
            {"sequence_value": 0.31, "amd_value": 0.78, "combined_value": 0.55}
        )
        self.assertEqual(a, c)

    def test_does_not_mutate_input(self):
        raw = {"sequence_value": 0.31, "amd_value": 0.78, "combined_value": 0.55}
        snapshot = dict(raw)
        _ = normalize_signal(raw)
        self.assertEqual(raw, snapshot)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            normalize_signal({"sequence_value": 0.5, "amd_value": 0.5})  # missing key
        with self.assertRaises(ValueError):
            normalize_signal(
                {"sequence_value": 1.5, "amd_value": 0.5, "combined_value": 0.5}
            )

    def test_int_inputs_are_explicitly_cast_to_float(self):
        # Integers 0 and 1 are mathematically in [0, 1]; they are explicitly
        # converted via float(). This is explicit casting, not implicit.
        out = normalize_signal(
            {"sequence_value": 1, "amd_value": 0, "combined_value": 1}
        )
        self.assertEqual(out["sequence_value"], 1.0)
        self.assertEqual(out["amd_value"], 0.0)
        self.assertEqual(out["combined_value"], 1.0)
        self.assertEqual(out["agreement"], 0.0)
        self.assertIsInstance(out["sequence_value"], float)
        self.assertIsInstance(out["amd_value"], float)
        self.assertIsInstance(out["combined_value"], float)
        self.assertIsInstance(out["agreement"], float)

    def test_rejects_bool_values(self):
        # Reinforces "no implicit casting" at the dict-input boundary.
        with self.assertRaises(ValueError):
            normalize_signal(
                {"sequence_value": True, "amd_value": 0.5, "combined_value": 0.5}
            )


if __name__ == "__main__":
    unittest.main()
