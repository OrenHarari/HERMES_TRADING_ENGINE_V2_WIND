"""Tests for Prompt 2 / Step 6C - Sample Size Hardening."""

import unittest

from hermes.learning.attribution import (
    AttributionConfig,
    compute_attribution,
)
from hermes.learning.sample_size import (
    DEFAULT_MIN_TRADES_PER_BUCKET,
    DEFAULT_MIN_TRADES_PER_COMBINATION,
    DEFAULT_MIN_TRADES_PER_REGIME,
    DEFAULT_MIN_TRADES_TOTAL,
    REASON_INSUFFICIENT_SAMPLE_SIZE,
    SampleSizeConfig,
    adapt_thresholds_safely,
    check_sample_size,
)
from hermes.learning.threshold_adapter import ThresholdAdapter


def _attribution_with_lower_eligible_bucket():
    return {
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


# -------- defaults match spec --------

class TestProductionDefaultsMatchSpec(unittest.TestCase):
    def test_min_trades_total_is_100(self):
        self.assertEqual(DEFAULT_MIN_TRADES_TOTAL, 100)
        self.assertEqual(SampleSizeConfig().min_trades_total, 100)

    def test_min_trades_per_bucket_is_20(self):
        self.assertEqual(DEFAULT_MIN_TRADES_PER_BUCKET, 20)
        self.assertEqual(SampleSizeConfig().min_trades_per_bucket, 20)

    def test_min_trades_per_regime_is_30(self):
        self.assertEqual(DEFAULT_MIN_TRADES_PER_REGIME, 30)
        self.assertEqual(SampleSizeConfig().min_trades_per_regime, 30)

    def test_min_trades_per_combination_is_50(self):
        self.assertEqual(DEFAULT_MIN_TRADES_PER_COMBINATION, 50)
        self.assertEqual(SampleSizeConfig().min_trades_per_combination, 50)


# -------- minimums are configurable, not hardcoded --------

class TestMinimumsAreConfigurable(unittest.TestCase):
    def test_test_injected_minimums_do_not_alter_production_defaults(self):
        # Test config with smaller minimums.
        small = SampleSizeConfig(
            min_trades_total=5,
            min_trades_per_bucket=2,
            min_trades_per_regime=2,
            min_trades_per_combination=2,
        )
        self.assertEqual(small.min_trades_total, 5)
        # Production default unchanged.
        self.assertEqual(SampleSizeConfig().min_trades_total, 100)

    def test_attribution_config_minimums_are_configurable(self):
        cfg = AttributionConfig(
            min_trades_per_bucket=2,
            min_trades_per_regime=2,
            min_trades_per_combination=2,
        )
        self.assertEqual(cfg.min_trades_per_bucket, 2)
        # Production default unchanged.
        self.assertEqual(AttributionConfig().min_trades_per_bucket, 20)


# -------- check_sample_size --------

class TestCheckSampleSize(unittest.TestCase):
    def test_below_minimum_total_returns_insufficient(self):
        out = check_sample_size(total_trades=50)
        self.assertFalse(out["sample_size_ok"])
        self.assertEqual(out["reason"], REASON_INSUFFICIENT_SAMPLE_SIZE)

    def test_at_minimum_total_returns_ok(self):
        out = check_sample_size(total_trades=100)
        self.assertTrue(out["sample_size_ok"])
        self.assertEqual(out["reason"], "")

    def test_above_minimum_total_returns_ok(self):
        out = check_sample_size(total_trades=500)
        self.assertTrue(out["sample_size_ok"])

    def test_custom_minimum_changes_threshold(self):
        cfg = SampleSizeConfig(min_trades_total=10)
        out_low = check_sample_size(total_trades=5, config=cfg)
        out_at = check_sample_size(total_trades=10, config=cfg)
        self.assertFalse(out_low["sample_size_ok"])
        self.assertTrue(out_at["sample_size_ok"])

    def test_output_shape(self):
        out = check_sample_size(total_trades=50)
        self.assertEqual(
            set(out.keys()), {"sample_size_ok", "reason", "details"}
        )
        self.assertIsInstance(out["sample_size_ok"], bool)
        self.assertIsInstance(out["reason"], str)
        self.assertIsInstance(out["details"], dict)
        self.assertIn("total_trades", out["details"])
        self.assertIn("required", out["details"])

    def test_rejects_negative_total(self):
        with self.assertRaises(ValueError):
            check_sample_size(total_trades=-1)

    def test_rejects_non_int_total(self):
        with self.assertRaises(ValueError):
            check_sample_size(total_trades="100")


# -------- adapt_thresholds_safely wrapper --------

class TestAdaptThresholdsSafely(unittest.TestCase):
    def test_below_total_threshold_returns_insufficient_sample_size(self):
        adapter = ThresholdAdapter()
        out = adapt_thresholds_safely(
            adapter,
            _attribution_with_lower_eligible_bucket(),
            total_trades=50,
            active_thresholds={"min_confidence": 0.70, "allow_chop": False},
        )
        self.assertFalse(out["thresholds_adapted"])
        self.assertEqual(out["reason"], REASON_INSUFFICIENT_SAMPLE_SIZE)

    def test_above_threshold_delegates_to_adapter(self):
        adapter = ThresholdAdapter()
        out = adapt_thresholds_safely(
            adapter,
            _attribution_with_lower_eligible_bucket(),
            total_trades=200,
            active_thresholds={"min_confidence": 0.70, "allow_chop": False},
        )
        # Adapter should now apply directly (non-addendum mode).
        self.assertTrue(out["thresholds_adapted"])

    def test_addendum_mode_above_threshold_creates_candidate(self):
        adapter = ThresholdAdapter(safety_addendum_active=True)
        out = adapt_thresholds_safely(
            adapter,
            _attribution_with_lower_eligible_bucket(),
            total_trades=200,
            active_thresholds={"min_confidence": 0.70, "allow_chop": False},
        )
        self.assertFalse(out["thresholds_adapted"])
        self.assertTrue(out["candidate_thresholds_created"])

    def test_wrapper_preserves_all_phase1_output_keys(self):
        adapter = ThresholdAdapter()
        out = adapt_thresholds_safely(
            adapter,
            _attribution_with_lower_eligible_bucket(),
            total_trades=50,
            active_thresholds={"min_confidence": 0.70, "allow_chop": False},
        )
        for k in (
            "thresholds_adapted",
            "candidate_thresholds_created",
            "reason",
            "active_thresholds_after",
            "proposals",
            "log",
        ):
            self.assertIn(k, out)

    def test_below_threshold_does_not_modify_active(self):
        active_before = {"min_confidence": 0.70, "allow_chop": False}
        adapter = ThresholdAdapter()
        out = adapt_thresholds_safely(
            adapter,
            _attribution_with_lower_eligible_bucket(),
            total_trades=50,
            active_thresholds=active_before,
        )
        self.assertEqual(out["active_thresholds_after"], active_before)


# -------- "no attribution below minimum sample size" --------

class TestAttributionMinimumsEnforced(unittest.TestCase):
    def test_default_bucket_minimum_excludes_small_samples(self):
        # 5 trades in a single bucket is far below default 20.
        trades = [
            {"confidence": 0.65, "regime": "trend_up", "pnl": 1.0}
            for _ in range(5)
        ]
        out = compute_attribution(trades)  # default config
        self.assertEqual(out["by_bucket"], [])

    def test_default_combination_minimum_excludes_below_50(self):
        # 49 trades in one bucket+regime cell -> default min_combination=50
        trades = [
            {"confidence": 0.65, "regime": "trend_up", "pnl": 1.0}
            for _ in range(49)
        ]
        out = compute_attribution(trades)
        self.assertEqual(out["by_combination"], [])


# -------- "insufficient sample size produces clear reason" --------

class TestClearReasonOnInsufficientSampleSize(unittest.TestCase):
    def test_reason_is_canonical_string(self):
        adapter = ThresholdAdapter()
        out = adapt_thresholds_safely(
            adapter,
            _attribution_with_lower_eligible_bucket(),
            total_trades=10,
            active_thresholds={"min_confidence": 0.70, "allow_chop": False},
        )
        # Spec-canonical reason.
        self.assertEqual(out["reason"], "insufficient_sample_size")

    def test_reason_is_documented_in_log(self):
        adapter = ThresholdAdapter()
        out = adapt_thresholds_safely(
            adapter,
            _attribution_with_lower_eligible_bucket(),
            total_trades=10,
            active_thresholds={"min_confidence": 0.70, "allow_chop": False},
        )
        # Log should include an entry attributing the rejection.
        self.assertTrue(out["log"])
        self.assertIn(
            "sample_size",
            " ".join(str(e) for e in out["log"]).lower(),
        )


if __name__ == "__main__":
    unittest.main()
