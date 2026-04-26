"""Tests for Prompt 1 / Step 2 - Orchestrator Consistency."""

import unittest

from hermes.orchestrator import (
    LABEL_NEUTRAL,
    LABEL_STRONG,
    LABEL_VALUES,
    LABEL_WEAK,
    REQUIRED_OUTPUT_KEYS,
    build_signal_output,
    derive_label,
)


class TestDeriveLabel(unittest.TestCase):
    def test_weak_below_lower_band(self):
        for v in (0.0, 0.1, 0.25, 0.39, 0.3999999):
            self.assertEqual(derive_label(v), LABEL_WEAK)

    def test_neutral_in_middle_band(self):
        for v in (0.4, 0.45, 0.5, 0.55, 0.5999999):
            self.assertEqual(derive_label(v), LABEL_NEUTRAL)

    def test_strong_at_or_above_upper_band(self):
        for v in (0.6, 0.7, 0.85, 0.99, 1.0):
            self.assertEqual(derive_label(v), LABEL_STRONG)

    def test_label_is_in_allowed_set(self):
        for v in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
            self.assertIn(derive_label(v), LABEL_VALUES)

    def test_deterministic(self):
        for v in (0.0, 0.3, 0.4, 0.55, 0.6, 0.99):
            self.assertEqual(derive_label(v), derive_label(v))


class TestBuildSignalOutput(unittest.TestCase):
    def _raw(self, seq=0.6, amd=0.4, comb=0.5):
        return {"sequence_value": seq, "amd_value": amd, "combined_value": comb}

    def test_required_keys_match_spec(self):
        # Step 2 spec lists exactly these output keys.
        self.assertEqual(
            set(REQUIRED_OUTPUT_KEYS),
            {"sequence_value", "amd_value", "combined_value", "agreement", "label"},
        )

    def test_returns_exact_key_set(self):
        out = build_signal_output(self._raw())
        self.assertEqual(set(out.keys()), set(REQUIRED_OUTPUT_KEYS))

    def test_no_missing_output_fields(self):
        out = build_signal_output(self._raw())
        for k in REQUIRED_OUTPUT_KEYS:
            self.assertIn(k, out)

    def test_numeric_values_in_unit_interval(self):
        for raw in [
            self._raw(0.0, 1.0, 0.5),
            self._raw(1.0, 0.0, 1.0),
            self._raw(0.123, 0.456, 0.789),
            self._raw(0.5, 0.5, 0.0),
        ]:
            out = build_signal_output(raw)
            for k in ("sequence_value", "amd_value", "combined_value", "agreement"):
                self.assertGreaterEqual(out[k], 0.0)
                self.assertLessEqual(out[k], 1.0)

    def test_agreement_correctness(self):
        out = build_signal_output(self._raw(0.7, 0.3, 0.5))
        self.assertAlmostEqual(out["agreement"], 0.6, places=12)

        out = build_signal_output(self._raw(0.0, 1.0, 0.5))
        self.assertAlmostEqual(out["agreement"], 0.0, places=12)

        out = build_signal_output(self._raw(0.42, 0.42, 0.5))
        self.assertAlmostEqual(out["agreement"], 1.0, places=12)

    def test_label_is_valid(self):
        for raw in [
            self._raw(comb=0.0),
            self._raw(comb=0.4),
            self._raw(comb=0.6),
            self._raw(comb=1.0),
        ]:
            out = build_signal_output(raw)
            self.assertIn(out["label"], LABEL_VALUES)

    def test_label_derived_only_from_combined_value(self):
        # Same combined_value -> same label, regardless of seq/amd values.
        labels = set()
        for seq in (0.0, 0.25, 0.5, 0.75, 1.0):
            for amd in (0.0, 0.25, 0.5, 0.75, 1.0):
                out = build_signal_output(
                    {"sequence_value": seq, "amd_value": amd, "combined_value": 0.5}
                )
                labels.add(out["label"])
        self.assertEqual(labels, {LABEL_NEUTRAL})

        labels = set()
        for seq in (0.0, 0.5, 1.0):
            for amd in (0.0, 0.5, 1.0):
                out = build_signal_output(
                    {"sequence_value": seq, "amd_value": amd, "combined_value": 0.85}
                )
                labels.add(out["label"])
        self.assertEqual(labels, {LABEL_STRONG})

    def test_label_at_exact_thresholds_is_deterministic(self):
        # combined_value == 0.4 -> neutral, == 0.6 -> strong (defined by spec).
        self.assertEqual(build_signal_output(self._raw(comb=0.4))["label"], LABEL_NEUTRAL)
        self.assertEqual(build_signal_output(self._raw(comb=0.6))["label"], LABEL_STRONG)
        self.assertEqual(build_signal_output(self._raw(comb=0.0))["label"], LABEL_WEAK)
        self.assertEqual(build_signal_output(self._raw(comb=1.0))["label"], LABEL_STRONG)

    def test_deterministic_same_input_same_output(self):
        raw = self._raw(0.31, 0.78, 0.55)
        a = build_signal_output(raw)
        b = build_signal_output(raw)
        c = build_signal_output(self._raw(0.31, 0.78, 0.55))
        self.assertEqual(a, b)
        self.assertEqual(b, c)

    def test_does_not_mutate_input(self):
        raw = self._raw(0.31, 0.78, 0.55)
        snapshot = dict(raw)
        _ = build_signal_output(raw)
        self.assertEqual(raw, snapshot)

    def test_no_external_mutable_state(self):
        # Calling 1000 times in any order with the same input must yield the
        # same dict; nothing accumulates between calls.
        raw = self._raw(0.42, 0.17, 0.66)
        first = build_signal_output(raw)
        for _ in range(1000):
            self.assertEqual(build_signal_output(raw), first)

    def test_rejects_invalid_input(self):
        # Validation is delegated to the signals layer; orchestrator must surface
        # the error, not silently coerce.
        with self.assertRaises(ValueError):
            build_signal_output({"sequence_value": 0.5, "amd_value": 0.5})  # missing key
        with self.assertRaises(ValueError):
            build_signal_output(
                {"sequence_value": 1.5, "amd_value": 0.5, "combined_value": 0.5}
            )
        with self.assertRaises(ValueError):
            build_signal_output(
                {"sequence_value": True, "amd_value": 0.5, "combined_value": 0.5}
            )

    def test_numeric_values_are_float(self):
        out = build_signal_output(self._raw(1, 0, 1))
        for k in ("sequence_value", "amd_value", "combined_value", "agreement"):
            self.assertIsInstance(out[k], float)
        self.assertIsInstance(out["label"], str)


if __name__ == "__main__":
    unittest.main()
