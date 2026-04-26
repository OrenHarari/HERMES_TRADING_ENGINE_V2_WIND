"""Tests for Prompt 1 / Step 6 Part 3 - Threshold Adaptation Baseline."""

import unittest

from hermes.learning.threshold_adapter import (
    THRESHOLD_BOUNDS,
    ThresholdAdapter,
)


def _attribution_with_buckets(buckets, regimes=None):
    """`buckets` is list of (lo, hi, win_rate, trade_count). regimes is dict
    of regime -> (win_rate, trade_count).
    """
    by_bucket = []
    for lo, hi, wr, tc in buckets:
        by_bucket.append(
            {
                "condition": "confidence={:.1f}-{:.1f}".format(lo, hi),
                "win_rate": wr,
                "trade_count": tc,
                "avg_net_pnl": 0.0,
                "profit_factor": 0.0,
            }
        )
    by_regime = []
    if regimes:
        for r, (wr, tc) in regimes.items():
            by_regime.append(
                {
                    "condition": "regime={!s}".format(r),
                    "win_rate": wr,
                    "trade_count": tc,
                    "avg_net_pnl": 0.0,
                    "profit_factor": 0.0,
                }
            )
    return {"by_bucket": by_bucket, "by_regime": by_regime, "by_combination": []}


class TestRefuseBelow100(unittest.TestCase):
    def test_returns_insufficient_when_under_100(self):
        a = ThresholdAdapter()
        attrib = _attribution_with_buckets([(0.6, 0.7, 0.99, 200)])
        result = a.propose(
            attrib, total_trades=99,
            active_thresholds={"min_confidence": 0.60, "allow_chop": False},
        )
        self.assertFalse(result["thresholds_adapted"])
        self.assertFalse(result["candidate_thresholds_created"])
        self.assertEqual(result["reason"], "insufficient_total_trades")

    def test_does_not_modify_active_when_insufficient(self):
        a = ThresholdAdapter()
        attrib = _attribution_with_buckets([(0.6, 0.7, 0.99, 200)])
        active = {"min_confidence": 0.60, "allow_chop": False}
        result = a.propose(attrib, total_trades=50, active_thresholds=active)
        self.assertEqual(result["active_thresholds_after"], active)


class TestApplyDirectlyPhase1(unittest.TestCase):
    def test_lowers_min_confidence_when_lower_bucket_meets_target(self):
        a = ThresholdAdapter()  # safety_addendum_active=False
        attrib = _attribution_with_buckets(
            [
                (0.5, 0.6, 0.60, 100),  # eligible & lowest
                (0.6, 0.7, 0.70, 100),
                (0.7, 0.8, 0.80, 100),
            ]
        )
        active = {"min_confidence": 0.70, "allow_chop": False}
        result = a.propose(attrib, total_trades=300, active_thresholds=active)
        self.assertTrue(result["thresholds_adapted"])
        self.assertFalse(result["candidate_thresholds_created"])
        self.assertEqual(
            result["active_thresholds_after"]["min_confidence"], 0.5
        )

    def test_no_change_when_proposal_equals_current(self):
        a = ThresholdAdapter()
        attrib = _attribution_with_buckets(
            [(0.7, 0.8, 0.99, 100)]
        )
        active = {"min_confidence": 0.70, "allow_chop": False}
        result = a.propose(attrib, total_trades=300, active_thresholds=active)
        self.assertFalse(result["thresholds_adapted"])

    def test_disallows_chop_when_chop_winrate_low(self):
        a = ThresholdAdapter()
        attrib = _attribution_with_buckets(
            [], regimes={"chop": (0.30, 100)}
        )
        active = {"min_confidence": 0.60, "allow_chop": True}
        result = a.propose(attrib, total_trades=300, active_thresholds=active)
        self.assertEqual(result["active_thresholds_after"]["allow_chop"], False)
        self.assertTrue(result["thresholds_adapted"])

    def test_allows_chop_when_chop_winrate_high(self):
        a = ThresholdAdapter()
        attrib = _attribution_with_buckets(
            [], regimes={"chop": (0.65, 100)}
        )
        active = {"min_confidence": 0.60, "allow_chop": False}
        result = a.propose(attrib, total_trades=300, active_thresholds=active)
        self.assertEqual(result["active_thresholds_after"]["allow_chop"], True)
        self.assertTrue(result["thresholds_adapted"])


class TestBounds(unittest.TestCase):
    def test_min_confidence_is_clamped_to_lower_bound(self):
        a = ThresholdAdapter()
        # Bucket 0.0-0.1 meets target -> proposed 0.0, must clamp to 0.40.
        attrib = _attribution_with_buckets([(0.0, 0.1, 0.99, 100)])
        active = {"min_confidence": 0.60, "allow_chop": False}
        result = a.propose(attrib, total_trades=300, active_thresholds=active)
        self.assertEqual(
            result["active_thresholds_after"]["min_confidence"],
            THRESHOLD_BOUNDS["min_confidence"][0],
        )

    def test_min_confidence_is_clamped_to_upper_bound(self):
        a = ThresholdAdapter()
        # Bucket 0.9-1.0 meets target -> proposed 0.90 (already at cap).
        attrib = _attribution_with_buckets([(0.9, 1.0, 0.99, 100)])
        active = {"min_confidence": 0.60, "allow_chop": False}
        result = a.propose(attrib, total_trades=300, active_thresholds=active)
        self.assertLessEqual(
            result["active_thresholds_after"]["min_confidence"],
            THRESHOLD_BOUNDS["min_confidence"][1],
        )


class TestAddendumMode(unittest.TestCase):
    def test_addendum_mode_does_not_apply_proposals(self):
        a = ThresholdAdapter(safety_addendum_active=True)
        attrib = _attribution_with_buckets(
            [(0.5, 0.6, 0.60, 100)]
        )
        active = {"min_confidence": 0.70, "allow_chop": False}
        result = a.propose(attrib, total_trades=300, active_thresholds=active)
        self.assertFalse(result["thresholds_adapted"])
        self.assertTrue(result["candidate_thresholds_created"])
        # Active untouched.
        self.assertEqual(result["active_thresholds_after"], active)
        # Proposal recorded.
        self.assertIn("min_confidence", result["proposals"])

    def test_addendum_no_change_when_no_proposal(self):
        a = ThresholdAdapter(safety_addendum_active=True)
        attrib = _attribution_with_buckets([(0.7, 0.8, 0.99, 100)])
        active = {"min_confidence": 0.70, "allow_chop": False}
        result = a.propose(attrib, total_trades=300, active_thresholds=active)
        self.assertFalse(result["thresholds_adapted"])
        self.assertFalse(result["candidate_thresholds_created"])


class TestRiskGuardrailsNotOverridden(unittest.TestCase):
    """Threshold adaptation must not produce values that override risk caps."""

    def test_min_confidence_cannot_drop_below_lower_bound(self):
        a = ThresholdAdapter()
        attrib = _attribution_with_buckets([(0.0, 0.1, 1.0, 1000)])
        active = {"min_confidence": 0.80, "allow_chop": False}
        result = a.propose(attrib, total_trades=300, active_thresholds=active)
        self.assertGreaterEqual(
            result["active_thresholds_after"]["min_confidence"], 0.40
        )


class TestDeterministic(unittest.TestCase):
    def test_same_input_same_output(self):
        a = ThresholdAdapter()
        attrib = _attribution_with_buckets(
            [(0.5, 0.6, 0.60, 100), (0.7, 0.8, 0.80, 100)]
        )
        active = {"min_confidence": 0.70, "allow_chop": False}
        r1 = a.propose(attrib, total_trades=300, active_thresholds=active)
        r2 = a.propose(attrib, total_trades=300, active_thresholds=active)
        self.assertEqual(r1, r2)


class TestRejectsBadInputs(unittest.TestCase):
    def test_active_thresholds_must_be_dict(self):
        a = ThresholdAdapter()
        with self.assertRaises(ValueError):
            a.propose({}, total_trades=100, active_thresholds="not dict")

    def test_active_must_have_required_keys(self):
        a = ThresholdAdapter()
        with self.assertRaises(ValueError):
            a.propose(
                {"by_bucket": [], "by_regime": []},
                total_trades=100,
                active_thresholds={"min_confidence": 0.6},
            )

    def test_constructor_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            ThresholdAdapter(target_win_rate=1.5)
        with self.assertRaises(ValueError):
            ThresholdAdapter(min_trades_for_adaptation=-1)
        with self.assertRaises(ValueError):
            ThresholdAdapter(safety_addendum_active="yes")
        with self.assertRaises(ValueError):
            ThresholdAdapter(
                chop_disallow_threshold=0.6, chop_allow_threshold=0.5
            )


if __name__ == "__main__":
    unittest.main()
