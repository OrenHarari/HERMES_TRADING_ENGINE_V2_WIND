"""Tests for Prompt 2 / Step 6B - Candidate Thresholds."""

import json
import os
import shutil
import tempfile
import unittest

from hermes.learning.candidate_thresholds import (
    DEFAULT_THRESHOLD_SCHEMA_VERSION,
    REASON_CANDIDATE_REJECTED_OOS,
    REASON_EDGE_DECAY_ACTIVE,
    REASON_KILL_SWITCH_ACTIVE,
    REASON_NO_CANDIDATE,
    REASON_OUT_OF_BOUNDS,
    REASON_PROMOTED,
    ThresholdStore,
    promote_candidate,
    propose_candidate,
)
from hermes.learning.threshold_adapter import ThresholdAdapter


def _attribution_with_lower_eligible_bucket():
    """Return an attribution_result where bucket 0.5-0.6 meets target."""
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


# ---------- ThresholdStore I/O ----------

class TestThresholdStoreFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_thr_")
        self.store = ThresholdStore(base_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_required_filenames(self):
        # Spec requires three files: active, candidate, adaptation log.
        self.assertEqual(
            os.path.basename(self.store.active_path), "active_thresholds.json"
        )
        self.assertEqual(
            os.path.basename(self.store.candidate_path),
            "candidate_thresholds.json",
        )
        self.assertEqual(
            os.path.basename(self.store.log_path),
            "threshold_adaptation_log.json",
        )

    def test_load_active_returns_defaults_when_missing(self):
        active = self.store.load_active()
        self.assertIn("min_confidence", active)
        self.assertIn("allow_chop", active)

    def test_load_candidate_returns_none_when_missing(self):
        self.assertIsNone(self.store.load_candidate())

    def test_save_and_load_active_round_trip(self):
        self.store.save_active(
            {"min_confidence": 0.55, "allow_chop": False},
            log_entry={"event": "manual_set"},
        )
        loaded = self.store.load_active()
        self.assertEqual(loaded["min_confidence"], 0.55)
        self.assertEqual(loaded["allow_chop"], False)

    def test_active_thresholds_schema_backward_compatible(self):
        # The active dict must contain the Phase-1 required keys.
        active = self.store.load_active()
        for k in ("min_confidence", "allow_chop"):
            self.assertIn(k, active)


# ---------- propose_candidate ----------

class TestProposeCandidate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_thr_")
        self.store = ThresholdStore(base_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_proposals_means_no_candidate(self):
        adaptation_result = {
            "thresholds_adapted": False,
            "candidate_thresholds_created": False,
            "proposals": {},
            "active_thresholds_after": self.store.load_active(),
            "log": [],
            "reason": "no_change_needed",
        }
        out = propose_candidate(adaptation_result, store=self.store)
        self.assertFalse(out["candidate_thresholds_created"])
        self.assertIsNone(self.store.load_candidate())

    def test_proposals_create_candidate_file(self):
        adaptation_result = {
            "thresholds_adapted": False,
            "candidate_thresholds_created": True,
            "proposals": {"min_confidence": 0.55},
            "active_thresholds_after": self.store.load_active(),
            "log": [],
            "reason": "candidate_only_addendum_active",
        }
        out = propose_candidate(adaptation_result, store=self.store)
        self.assertTrue(out["candidate_thresholds_created"])
        cand = self.store.load_candidate()
        self.assertIsNotNone(cand)
        self.assertEqual(cand["min_confidence"], 0.55)

    def test_propose_does_not_modify_active(self):
        active_before = self.store.load_active()
        adaptation_result = {
            "thresholds_adapted": False,
            "candidate_thresholds_created": True,
            "proposals": {"min_confidence": 0.55},
            "active_thresholds_after": active_before,
            "log": [],
            "reason": "candidate_only_addendum_active",
        }
        propose_candidate(adaptation_result, store=self.store)
        active_after = self.store.load_active()
        self.assertEqual(active_before, active_after)

    def test_propose_logs_event(self):
        adaptation_result = {
            "thresholds_adapted": False,
            "candidate_thresholds_created": True,
            "proposals": {"min_confidence": 0.55},
            "active_thresholds_after": self.store.load_active(),
            "log": [],
            "reason": "candidate_only_addendum_active",
        }
        propose_candidate(adaptation_result, store=self.store)
        log = self.store.get_log()
        self.assertTrue(any(e.get("event") == "candidate_proposed" for e in log))

    def test_propose_blocks_out_of_bounds(self):
        # min_confidence <= bound is blocked.
        adaptation_result = {
            "thresholds_adapted": False,
            "candidate_thresholds_created": True,
            "proposals": {"min_confidence": 0.10},  # below 0.40 bound
            "active_thresholds_after": self.store.load_active(),
            "log": [],
            "reason": "candidate_only_addendum_active",
        }
        out = propose_candidate(adaptation_result, store=self.store)
        self.assertFalse(out["candidate_thresholds_created"])
        self.assertEqual(out["reason"], REASON_OUT_OF_BOUNDS)
        self.assertIsNone(self.store.load_candidate())


# ---------- promote_candidate ----------

class TestPromoteCandidate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_thr_")
        self.store = ThresholdStore(base_dir=self.tmp)
        # Stage a candidate.
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

    def test_no_candidate_means_no_promotion(self):
        # Clear candidate first.
        self.store.clear_candidate(log_entry={"event": "manual_clear"})
        out = promote_candidate(
            store=self.store,
            validation_result={"validation_passed": True, "reason": ""},
        )
        self.assertFalse(out["thresholds_adapted"])
        self.assertFalse(out["candidate_thresholds_promoted"])
        self.assertEqual(out["reason"], REASON_NO_CANDIDATE)

    def test_failed_validation_keeps_active_unchanged(self):
        active_before = self.store.load_active()
        out = promote_candidate(
            store=self.store,
            validation_result={
                "validation_passed": False, "reason": "weak_oos",
            },
        )
        self.assertFalse(out["thresholds_adapted"])
        self.assertFalse(out["candidate_thresholds_promoted"])
        self.assertEqual(out["reason"], REASON_CANDIDATE_REJECTED_OOS)
        self.assertEqual(self.store.load_active(), active_before)

    def test_failed_validation_logs_rejection(self):
        promote_candidate(
            store=self.store,
            validation_result={
                "validation_passed": False, "reason": "weak_oos",
            },
        )
        log = self.store.get_log()
        self.assertTrue(any(
            e.get("event") == "candidate_rejected" for e in log
        ))

    def test_edge_decay_blocks_promotion(self):
        active_before = self.store.load_active()
        out = promote_candidate(
            store=self.store,
            validation_result={"validation_passed": True, "reason": ""},
            edge_decay_alert=True,
        )
        self.assertFalse(out["candidate_thresholds_promoted"])
        self.assertEqual(out["reason"], REASON_EDGE_DECAY_ACTIVE)
        self.assertEqual(self.store.load_active(), active_before)

    def test_kill_switch_blocks_promotion(self):
        active_before = self.store.load_active()
        out = promote_candidate(
            store=self.store,
            validation_result={"validation_passed": True, "reason": ""},
            kill_switch_active=True,
        )
        self.assertFalse(out["candidate_thresholds_promoted"])
        self.assertEqual(out["reason"], REASON_KILL_SWITCH_ACTIVE)
        self.assertEqual(self.store.load_active(), active_before)

    def test_successful_promotion_updates_active_and_clears_candidate(self):
        out = promote_candidate(
            store=self.store,
            validation_result={"validation_passed": True, "reason": ""},
        )
        self.assertTrue(out["thresholds_adapted"])
        self.assertTrue(out["candidate_thresholds_promoted"])
        self.assertEqual(out["reason"], REASON_PROMOTED)
        active = self.store.load_active()
        self.assertEqual(active["min_confidence"], 0.55)
        # Candidate should be cleared after promotion.
        self.assertIsNone(self.store.load_candidate())

    def test_promotion_logs_event(self):
        promote_candidate(
            store=self.store,
            validation_result={"validation_passed": True, "reason": ""},
        )
        log = self.store.get_log()
        self.assertTrue(any(e.get("event") == "candidate_promoted" for e in log))

    def test_thresholds_remain_bounded_after_promotion(self):
        # After promotion, active min_confidence must still be within bounds.
        promote_candidate(
            store=self.store,
            validation_result={"validation_passed": True, "reason": ""},
        )
        active = self.store.load_active()
        self.assertGreaterEqual(active["min_confidence"], 0.40)
        self.assertLessEqual(active["min_confidence"], 0.90)


# ---------- candidate thresholds do not affect active trading ----------

class TestCandidateDoesNotAffectActive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_thr_")
        self.store = ThresholdStore(base_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_candidate_present_but_active_unchanged_until_promotion(self):
        active_before = self.store.load_active()
        propose_candidate(
            {
                "thresholds_adapted": False,
                "candidate_thresholds_created": True,
                "proposals": {"min_confidence": 0.85},
                "active_thresholds_after": active_before,
                "log": [],
                "reason": "candidate_only_addendum_active",
            },
            store=self.store,
        )
        # Active not affected.
        self.assertEqual(self.store.load_active(), active_before)
        # Candidate exists.
        self.assertIsNotNone(self.store.load_candidate())


# ---------- direct updates blocked when addendum is active ----------

class TestAddendumBlocksDirectUpdates(unittest.TestCase):
    """Spec: direct threshold updates are blocked when the addendum is active.

    The Phase 1 adapter already supports a `safety_addendum_active=True` mode
    which creates candidates instead of applying. We re-verify here from the
    Step 6B viewpoint.
    """

    def test_addendum_mode_does_not_apply_directly(self):
        adapter = ThresholdAdapter(safety_addendum_active=True)
        result = adapter.propose(
            _attribution_with_lower_eligible_bucket(),
            total_trades=200,
            active_thresholds={"min_confidence": 0.70, "allow_chop": False},
        )
        self.assertFalse(result["thresholds_adapted"])
        self.assertTrue(result["candidate_thresholds_created"])
        self.assertEqual(
            result["active_thresholds_after"]["min_confidence"], 0.70
        )

    def test_old_step6_output_keys_remain_supported(self):
        adapter = ThresholdAdapter(safety_addendum_active=True)
        result = adapter.propose(
            _attribution_with_lower_eligible_bucket(),
            total_trades=200,
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
            self.assertIn(k, result)


# ---------- bounds + persistence sanity ----------

class TestBoundsAndPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_thr_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_log_persists_across_store_reopens(self):
        s1 = ThresholdStore(base_dir=self.tmp)
        propose_candidate(
            {
                "thresholds_adapted": False,
                "candidate_thresholds_created": True,
                "proposals": {"min_confidence": 0.55},
                "active_thresholds_after": s1.load_active(),
                "log": [],
                "reason": "candidate_only_addendum_active",
            },
            store=s1,
        )
        s2 = ThresholdStore(base_dir=self.tmp)
        log = s2.get_log()
        self.assertTrue(any(e.get("event") == "candidate_proposed" for e in log))

    def test_active_file_uses_schema_version(self):
        s = ThresholdStore(base_dir=self.tmp)
        s.save_active(
            {"min_confidence": 0.55, "allow_chop": False},
            log_entry={"event": "set"},
        )
        # File on disk should include a schema_version key.
        with open(s.active_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        self.assertEqual(
            blob.get("schema_version"), DEFAULT_THRESHOLD_SCHEMA_VERSION
        )


if __name__ == "__main__":
    unittest.main()
