"""Tests for Prompt 1 / Step 5 Part 1 - Confidence model."""

import unittest

from hermes.decision.confidence import (
    DEFAULT_REGIME_WEIGHTS,
    DEFAULT_WEIGHTS,
    compute_confidence_score,
    default_regime_weights,
)
from hermes.market import (
    REGIME_CHOP,
    REGIME_HIGH_VOLATILITY,
    REGIME_LOW_VOLATILITY,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
)


def _signal(seq=0.7, amd=0.7, comb=0.7, agreement=1.0):
    return {
        "sequence_value": seq,
        "amd_value": amd,
        "combined_value": comb,
        "agreement": agreement,
    }


def _intel(regime=REGIME_TREND_UP, vol=0.3, mom=0.7):
    return {"regime": regime, "volatility_score": vol, "momentum_score": mom}


class TestComputeConfidence(unittest.TestCase):
    def test_returns_float_in_unit_interval(self):
        score = compute_confidence_score(_signal(), _intel())
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_deterministic(self):
        a = compute_confidence_score(_signal(), _intel())
        b = compute_confidence_score(_signal(), _intel())
        self.assertEqual(a, b)

    def test_high_inputs_high_confidence(self):
        score = compute_confidence_score(
            _signal(seq=1.0, amd=1.0, comb=1.0, agreement=1.0),
            _intel(regime=REGIME_TREND_UP, vol=0.0, mom=1.0),
        )
        # Best case: should be at the high end.
        self.assertGreater(score, 0.85)

    def test_low_inputs_low_confidence(self):
        score = compute_confidence_score(
            _signal(seq=0.0, amd=0.0, comb=0.0, agreement=0.0),
            _intel(regime=REGIME_TREND_DOWN, vol=1.0, mom=0.0),
        )
        # Worst case: with these defaults, score should be 0.0.
        self.assertLess(score, 0.05)

    def test_trend_down_lower_than_trend_up_for_same_other_inputs(self):
        sig = _signal()
        a = compute_confidence_score(sig, _intel(regime=REGIME_TREND_UP))
        b = compute_confidence_score(sig, _intel(regime=REGIME_TREND_DOWN))
        self.assertGreater(a, b)

    def test_chop_lower_than_trend_up(self):
        sig = _signal()
        a = compute_confidence_score(sig, _intel(regime=REGIME_TREND_UP))
        b = compute_confidence_score(sig, _intel(regime=REGIME_CHOP))
        self.assertGreater(a, b)

    def test_high_volatility_penalizes_confidence(self):
        sig = _signal()
        a = compute_confidence_score(sig, _intel(regime=REGIME_LOW_VOLATILITY, vol=0.0, mom=0.5))
        b = compute_confidence_score(sig, _intel(regime=REGIME_HIGH_VOLATILITY, vol=1.0, mom=0.5))
        self.assertGreater(a, b)

    def test_default_regime_weights_returns_copy(self):
        a = default_regime_weights()
        a[REGIME_TREND_UP] = 0.0
        b = default_regime_weights()
        self.assertEqual(b[REGIME_TREND_UP], DEFAULT_REGIME_WEIGHTS[REGIME_TREND_UP])

    def test_rejects_bad_signal(self):
        bad = {
            "sequence_value": 0.5,
            "amd_value": 0.5,
            "combined_value": 0.5,
        }  # missing agreement
        with self.assertRaises(ValueError):
            compute_confidence_score(bad, _intel())

    def test_rejects_bad_intelligence(self):
        with self.assertRaises(ValueError):
            compute_confidence_score(_signal(), {"regime": "unknown", "volatility_score": 0.5, "momentum_score": 0.5})
        with self.assertRaises(ValueError):
            compute_confidence_score(_signal(), {"regime": REGIME_TREND_UP, "volatility_score": 1.5, "momentum_score": 0.5})

    def test_rejects_bad_weights(self):
        with self.assertRaises(ValueError):
            compute_confidence_score(
                _signal(),
                _intel(),
                weights={"signal": 0.5, "agreement": 0.5, "momentum": 0.0,
                        "volatility": 0.0, "regime": 0.5},  # sums to 1.5
            )

    def test_default_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(DEFAULT_WEIGHTS.values()), 1.0, places=12)

    def test_custom_regime_weights_used(self):
        # If we mark trend_down at 1.0, score for trend_down should rise.
        sig = _signal()
        intel = _intel(regime=REGIME_TREND_DOWN)
        default_score = compute_confidence_score(sig, intel)
        custom = dict(DEFAULT_REGIME_WEIGHTS)
        custom[REGIME_TREND_DOWN] = 1.0
        boosted_score = compute_confidence_score(sig, intel, regime_weights=custom)
        self.assertGreater(boosted_score, default_score)


if __name__ == "__main__":
    unittest.main()
