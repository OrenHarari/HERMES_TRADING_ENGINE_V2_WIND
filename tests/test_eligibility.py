"""Tests for Prompt 1 / Step 5 Part 2 - Trade Eligibility Gate."""

import unittest

from hermes.decision.config import DecisionConfig
from hermes.decision.eligibility import (
    REASON_LOW_AGREEMENT,
    REASON_LOW_CONFIDENCE,
    REASON_OK,
    REASON_REGIME_CHOP,
    REASON_VOL_HIGH,
    REASON_VOL_LOW,
    check_eligibility,
)
from hermes.market import (
    REGIME_CHOP,
    REGIME_HIGH_VOLATILITY,
    REGIME_LOW_VOLATILITY,
    REGIME_TREND_UP,
)


class TestCheckEligibility(unittest.TestCase):
    def _ok_args(self, **overrides):
        args = {
            "confidence": 0.80,
            "agreement": 0.80,
            "regime": REGIME_TREND_UP,
            "volatility_score": 0.5,
        }
        args.update(overrides)
        return args

    def test_passes_when_all_conditions_met(self):
        out = check_eligibility(**self._ok_args())
        self.assertTrue(out["trade_allowed"])
        self.assertEqual(out["reason_if_blocked"], REASON_OK)

    def test_blocks_low_confidence_first(self):
        out = check_eligibility(
            **self._ok_args(confidence=0.50, agreement=0.30, volatility_score=0.0)
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason_if_blocked"], REASON_LOW_CONFIDENCE)

    def test_blocks_low_agreement(self):
        out = check_eligibility(**self._ok_args(agreement=0.30))
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason_if_blocked"], REASON_LOW_AGREEMENT)

    def test_blocks_chop_when_disallowed(self):
        out = check_eligibility(**self._ok_args(regime=REGIME_CHOP))
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason_if_blocked"], REASON_REGIME_CHOP)

    def test_allows_chop_when_explicitly_allowed(self):
        cfg = DecisionConfig(allow_chop=True)
        out = check_eligibility(**self._ok_args(regime=REGIME_CHOP), config=cfg)
        self.assertTrue(out["trade_allowed"])

    def test_blocks_volatility_too_low(self):
        out = check_eligibility(**self._ok_args(volatility_score=0.0))
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason_if_blocked"], REASON_VOL_LOW)

    def test_blocks_volatility_too_high(self):
        out = check_eligibility(**self._ok_args(volatility_score=1.0))
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason_if_blocked"], REASON_VOL_HIGH)

    def test_high_confidence_does_not_override_chop_block(self):
        out = check_eligibility(**self._ok_args(confidence=1.0, regime=REGIME_CHOP))
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason_if_blocked"], REASON_REGIME_CHOP)

    def test_at_threshold_boundaries_is_allowed(self):
        cfg = DecisionConfig()
        out = check_eligibility(
            confidence=cfg.min_confidence,
            agreement=cfg.min_agreement,
            regime=REGIME_LOW_VOLATILITY,
            volatility_score=cfg.volatility_min,
            config=cfg,
        )
        # At the lower bound it's accepted (>= and >=).
        # But low_volatility regime is allowed (only chop is blocked).
        self.assertTrue(out["trade_allowed"])

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            check_eligibility(1.1, 0.8, REGIME_TREND_UP, 0.5)
        with self.assertRaises(ValueError):
            check_eligibility(0.8, 0.8, "unknown_regime", 0.5)
        with self.assertRaises(ValueError):
            check_eligibility(0.8, 0.8, REGIME_TREND_UP, -0.1)

    def test_high_volatility_regime_with_acceptable_score_not_blocked_by_regime(self):
        # The eligibility gate only treats CHOP specially. high_volatility
        # regime is allowed if the volatility_score is in band.
        out = check_eligibility(
            confidence=0.80, agreement=0.80,
            regime=REGIME_HIGH_VOLATILITY, volatility_score=0.50,
        )
        self.assertTrue(out["trade_allowed"])


if __name__ == "__main__":
    unittest.main()
