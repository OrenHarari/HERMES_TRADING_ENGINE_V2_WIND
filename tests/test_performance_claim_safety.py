"""Tests for Prompt 2 / Step 6E - Performance Claim Safety."""

import os
import shutil
import tempfile
import unittest

from hermes.learning.candidate_thresholds import (
    ThresholdStore,
    promote_candidate,
    propose_candidate,
)
from hermes.learning.oos_gate import evaluate_oos_promotion
from hermes.learning.performance_claim import (
    CLAIM_IMPROVED,
    CLAIM_INSUFFICIENT_EVIDENCE,
    CLAIM_PRESERVED,
    CLAIM_REJECTED,
    CLAIM_WORSENED,
    PerformanceClaimConfig,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_NO_VALIDATION_EVIDENCE,
    VALID_CLAIMS,
    evaluate_performance_claim,
)


def _report(
    trade_count=100,
    win_rate=0.55,
    profit_factor=1.5,
    max_drawdown=100.0,
    stability_score=0.50,
    net_pnl=500.0,
):
    return {
        "trade_count": trade_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "stability_score": stability_score,
        "net_pnl": net_pnl,
        "avg_win": 10.0, "avg_loss": -5.0,
        "trades_per_regime": {}, "cost_model_applied": True,
    }


def _windows():
    return ("2025-01", "2025-06"), ("2025-07", "2025-08")


def _passing_oos(active=None, candidate=None):
    d, v = _windows()
    return evaluate_oos_promotion(
        d, v, active or _report(), candidate or _report(),
    )


def _failing_oos(reason_active=None, reason_candidate=None):
    d, v = _windows()
    return evaluate_oos_promotion(
        d, v,
        reason_active or _report(profit_factor=2.0),
        reason_candidate or _report(profit_factor=0.5),
    )


# ---------- claim enumeration ----------

class TestClaimEnum(unittest.TestCase):
    def test_canonical_claim_values(self):
        self.assertEqual(
            set(VALID_CLAIMS),
            {
                CLAIM_IMPROVED,
                CLAIM_PRESERVED,
                CLAIM_WORSENED,
                CLAIM_REJECTED,
                CLAIM_INSUFFICIENT_EVIDENCE,
            },
        )


# ---------- "improvement claim must be supported by validation" ----------

class TestImprovementRequiresEvidence(unittest.TestCase):
    def test_no_oos_means_insufficient_evidence_even_if_metrics_better(self):
        active = _report(profit_factor=1.0, max_drawdown=100.0)
        candidate = _report(profit_factor=2.5, max_drawdown=50.0)
        out = evaluate_performance_claim(
            active_report=active,
            candidate_report=candidate,
            oos_validation_result=None,
        )
        self.assertEqual(out["claim"], CLAIM_INSUFFICIENT_EVIDENCE)
        self.assertEqual(out["reason"], REASON_NO_VALIDATION_EVIDENCE)

    def test_passed_oos_with_better_metrics_yields_improved(self):
        oos = _passing_oos()
        out = evaluate_performance_claim(
            active_report=_report(profit_factor=1.5, max_drawdown=100.0),
            candidate_report=_report(profit_factor=1.8, max_drawdown=80.0),
            oos_validation_result=oos,
        )
        self.assertEqual(out["claim"], CLAIM_IMPROVED)

    def test_passed_oos_with_no_material_change_yields_preserved(self):
        oos = _passing_oos()
        # Identical metrics -> preserved.
        out = evaluate_performance_claim(
            active_report=_report(),
            candidate_report=_report(),
            oos_validation_result=oos,
        )
        self.assertEqual(out["claim"], CLAIM_PRESERVED)


# ---------- "rejected candidates produce no improvement claim" ----------

class TestRejectedNeverImproved(unittest.TestCase):
    def test_failed_oos_yields_rejected_claim(self):
        oos = _failing_oos()
        # Even if the candidate has a higher win_rate, OOS rejected.
        out = evaluate_performance_claim(
            active_report=_report(win_rate=0.50),
            candidate_report=_report(win_rate=0.80, profit_factor=0.5),
            oos_validation_result=oos,
        )
        self.assertEqual(out["claim"], CLAIM_REJECTED)
        # The OOS rejection reason is preserved.
        self.assertEqual(out["reason"], oos["reason"])


# ---------- "win_rate alone is not enough" ----------

class TestWinRateAloneInsufficient(unittest.TestCase):
    def test_higher_win_rate_alone_does_not_yield_improved(self):
        oos = _passing_oos()
        # Same PF, same MDD, same stability; only win_rate is higher.
        out = evaluate_performance_claim(
            active_report=_report(win_rate=0.50, profit_factor=1.5,
                                  max_drawdown=100.0, stability_score=0.5),
            candidate_report=_report(win_rate=0.70, profit_factor=1.5,
                                     max_drawdown=100.0, stability_score=0.5),
            oos_validation_result=oos,
        )
        # Robustness metrics unchanged -> preserved (NOT improved).
        self.assertEqual(out["claim"], CLAIM_PRESERVED)

    def test_higher_win_rate_with_better_robustness_yields_improved(self):
        oos = _passing_oos()
        out = evaluate_performance_claim(
            active_report=_report(win_rate=0.50, profit_factor=1.5,
                                  max_drawdown=100.0),
            candidate_report=_report(win_rate=0.60, profit_factor=2.0,
                                     max_drawdown=80.0),
            oos_validation_result=oos,
        )
        self.assertEqual(out["claim"], CLAIM_IMPROVED)

    def test_lower_win_rate_with_better_robustness_yields_improved(self):
        # Spec: "A lower win_rate with better profit_factor and lower
        # drawdown may be acceptable."
        oos = _passing_oos()
        out = evaluate_performance_claim(
            active_report=_report(win_rate=0.60, profit_factor=1.2,
                                  max_drawdown=200.0),
            candidate_report=_report(win_rate=0.50, profit_factor=2.0,
                                     max_drawdown=100.0),
            oos_validation_result=oos,
        )
        self.assertEqual(out["claim"], CLAIM_IMPROVED)


# ---------- worsened path ----------

class TestWorsenedClaim(unittest.TestCase):
    def test_oos_passed_but_metrics_dipped_yields_worsened(self):
        oos = _passing_oos()
        # Both PF and MDD got worse, even though OOS gate (with its tolerance)
        # accepted the candidate. The claim must call this honestly.
        out = evaluate_performance_claim(
            active_report=_report(profit_factor=1.50, max_drawdown=100.0),
            # PF dropped from 1.50 to 1.43 (within OOS 0.95 tolerance), and
            # MDD rose from 100 to 115 (within OOS 1.20 tolerance).
            candidate_report=_report(profit_factor=1.43, max_drawdown=115.0),
            oos_validation_result=oos,
        )
        self.assertEqual(out["claim"], CLAIM_WORSENED)


# ---------- insufficient evidence ----------

class TestInsufficientEvidence(unittest.TestCase):
    def test_low_candidate_trade_count_yields_insufficient(self):
        oos = _passing_oos()
        cfg = PerformanceClaimConfig(min_evidence_trades=50)
        out = evaluate_performance_claim(
            active_report=_report(trade_count=200),
            candidate_report=_report(trade_count=10),
            oos_validation_result=oos,
            config=cfg,
        )
        self.assertEqual(out["claim"], CLAIM_INSUFFICIENT_EVIDENCE)
        self.assertEqual(out["reason"], REASON_INSUFFICIENT_EVIDENCE)


# ---------- output shape + determinism ----------

class TestOutputShape(unittest.TestCase):
    def test_canonical_keys(self):
        oos = _passing_oos()
        out = evaluate_performance_claim(
            _report(), _report(), oos
        )
        self.assertEqual(set(out.keys()), {"claim", "reason", "evidence"})
        self.assertIn(out["claim"], VALID_CLAIMS)

    def test_evidence_includes_metric_deltas(self):
        oos = _passing_oos()
        out = evaluate_performance_claim(
            _report(profit_factor=1.5),
            _report(profit_factor=1.8),
            oos,
        )
        self.assertIn("active_profit_factor", out["evidence"])
        self.assertIn("candidate_profit_factor", out["evidence"])

    def test_deterministic(self):
        oos = _passing_oos()
        a = evaluate_performance_claim(_report(), _report(), oos)
        b = evaluate_performance_claim(_report(), _report(), oos)
        self.assertEqual(a, b)


# ---------- "system does not force threshold updates" ----------

class TestNoForcedUpdates(unittest.TestCase):
    """Calling the evaluator must not trigger any state change in the
    threshold store; it is a pure observer.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_pcs_")
        self.store = ThresholdStore(base_dir=self.tmp)
        propose_candidate(
            {
                "thresholds_adapted": False,
                "candidate_thresholds_created": True,
                "proposals": {"min_confidence": 0.55},
                "active_thresholds_after": self.store.load_active(),
                "log": [],
                "reason": "candidate_only_addendum_active",
            },
            store=self.store,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_evaluator_does_not_promote_candidate(self):
        active_before = self.store.load_active()
        candidate_before = self.store.load_candidate()
        oos = _passing_oos()
        evaluate_performance_claim(_report(), _report(), oos)
        # No state change.
        self.assertEqual(self.store.load_active(), active_before)
        self.assertEqual(self.store.load_candidate(), candidate_before)

    def test_system_can_reject_after_many_losses(self):
        """Spec: 'system can reject adaptation even after many losses'.

        We simulate 'many losses' as a failed OOS validation; the claim
        must be 'rejected' and the active thresholds must remain unchanged
        when promote_candidate is then called with the failed validation
        result.
        """
        active_before = self.store.load_active()
        oos = _failing_oos()
        claim = evaluate_performance_claim(_report(), _report(), oos)
        self.assertEqual(claim["claim"], CLAIM_REJECTED)
        # Use the same OOS result as the validation_result for promotion.
        promotion = promote_candidate(self.store, validation_result=oos)
        self.assertFalse(promotion["candidate_thresholds_promoted"])
        self.assertEqual(self.store.load_active(), active_before)


if __name__ == "__main__":
    unittest.main()
