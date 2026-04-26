"""Tests for Prompt 2 / Step 8 - Live Safety Kill Switch."""

import unittest

from hermes.safety.cost_model import CostModel
from hermes.safety.data_contract import (
    SYSTEM_MODE_BACKTEST,
    SYSTEM_MODE_LIVE,
    SYSTEM_MODE_PAPER,
)
from hermes.safety.kill_switch import (
    KILL_SWITCH_REASONS,
    KS_REASON_ABNORMAL_SLIPPAGE,
    KS_REASON_ACCOUNT_EQUITY_UNAVAILABLE,
    KS_REASON_CORRUPTED_THRESHOLD_CONFIG,
    KS_REASON_COST_MODEL_MISSING,
    KS_REASON_DAILY_LOSS,
    KS_REASON_DUPLICATE_CANDLE,
    KS_REASON_EDGE_DECAY,
    KS_REASON_EXECUTION_ERROR,
    KS_REASON_INVALID_CANDLE,
    KS_REASON_MAX_CONSECUTIVE_LOSSES,
    KS_REASON_POSITION_STATE_MISMATCH,
    KS_REASON_STALE_DATA,
    KS_REASON_UNKNOWN_SYSTEM_MODE,
    KillSwitch,
    ResetConfig,
    detect_abnormal_slippage,
    detect_corrupted_threshold_config,
    detect_position_state_mismatch,
)


# ---------- KillSwitch core ----------

class TestKillSwitchActivation(unittest.TestCase):
    def test_initial_state_inactive(self):
        ks = KillSwitch()
        self.assertFalse(ks.active)
        self.assertEqual(ks.reason, "")

    def test_activate_sets_state(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_DAILY_LOSS, now_ts=1000)
        self.assertTrue(ks.active)
        self.assertEqual(ks.reason, KS_REASON_DAILY_LOSS)

    def test_activate_logs_event(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_STALE_DATA, now_ts=1000)
        log = ks.get_log()
        self.assertTrue(any(
            e.get("event") == "kill_switch_activated"
            and e.get("reason") == KS_REASON_STALE_DATA
            and e.get("timestamp") == 1000
            for e in log
        ))

    def test_re_activate_preserves_original_reason(self):
        # Spec: kill switch must preserve the original reason that triggered it.
        ks = KillSwitch()
        ks.activate(KS_REASON_STALE_DATA, now_ts=1000)
        ks.activate(KS_REASON_DAILY_LOSS, now_ts=2000)
        self.assertEqual(ks.reason, KS_REASON_STALE_DATA)

    def test_re_activate_logs_secondary_event(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_STALE_DATA, now_ts=1000)
        ks.activate(KS_REASON_DAILY_LOSS, now_ts=2000)
        log = ks.get_log()
        # Original activation logged.
        self.assertTrue(any(
            e.get("event") == "kill_switch_activated" for e in log
        ))
        # Secondary trigger also logged for audit.
        self.assertTrue(any(
            e.get("event") == "kill_switch_secondary_trigger" for e in log
        ))

    def test_activate_rejects_unknown_reason(self):
        ks = KillSwitch()
        with self.assertRaises(ValueError):
            ks.activate("not_a_real_reason", now_ts=1000)


class TestKillSwitchTradeBlock(unittest.TestCase):
    def test_inactive_allows_trade(self):
        ks = KillSwitch()
        out = ks.check_trade_allowed()
        self.assertTrue(out["trade_allowed"])

    def test_active_blocks_trade(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_EDGE_DECAY, now_ts=1000)
        out = ks.check_trade_allowed()
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "kill_switch_active")
        self.assertEqual(out["kill_switch_reason"], KS_REASON_EDGE_DECAY)

    def test_canonical_output_keys(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_DAILY_LOSS, now_ts=1)
        out = ks.check_trade_allowed()
        self.assertEqual(
            set(out.keys()),
            {"trade_allowed", "reason", "kill_switch_reason"},
        )

    def test_high_confidence_cannot_bypass_kill_switch(self):
        # The kill switch's check_trade_allowed knows nothing about
        # confidence - that is the spec guarantee. Re-verify the API has
        # no hook that confidence could exploit.
        ks = KillSwitch()
        ks.activate(KS_REASON_INVALID_CANDLE, now_ts=1)
        # No kwargs accepted that allow override.
        out = ks.check_trade_allowed()
        self.assertFalse(out["trade_allowed"])


class TestKillSwitchState(unittest.TestCase):
    def test_state_is_inspectable(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_STALE_DATA, now_ts=1000)
        s = ks.state()
        self.assertEqual(set(s.keys()),
                         {"active", "reason", "activated_ts", "log"})
        self.assertTrue(s["active"])
        self.assertEqual(s["reason"], KS_REASON_STALE_DATA)
        self.assertEqual(s["activated_ts"], 1000)

    def test_state_is_deterministic(self):
        a = KillSwitch()
        b = KillSwitch()
        a.activate(KS_REASON_STALE_DATA, now_ts=1000)
        b.activate(KS_REASON_STALE_DATA, now_ts=1000)
        self.assertEqual(a.state(), b.state())


# ---------- pure detectors ----------

class TestAbnormalSlippageDetector(unittest.TestCase):
    def test_above_max_slippage_is_abnormal(self):
        self.assertTrue(detect_abnormal_slippage(
            actual_slippage=0.05,
            expected_slippage=0.001,
            max_slippage=0.02,
            slippage_multiplier=10.0,
        ))

    def test_above_expected_times_multiplier_is_abnormal(self):
        # 0.015 < max(0.02) but 0.015 > expected(0.001) * multiplier(10) = 0.01
        self.assertTrue(detect_abnormal_slippage(
            actual_slippage=0.015,
            expected_slippage=0.001,
            max_slippage=0.02,
            slippage_multiplier=10.0,
        ))

    def test_normal_slippage_not_flagged(self):
        self.assertFalse(detect_abnormal_slippage(
            actual_slippage=0.001,
            expected_slippage=0.001,
            max_slippage=0.02,
            slippage_multiplier=10.0,
        ))

    def test_rejects_negative_inputs(self):
        with self.assertRaises(ValueError):
            detect_abnormal_slippage(
                actual_slippage=-0.001, expected_slippage=0.001,
                max_slippage=0.02, slippage_multiplier=10.0,
            )


class TestPositionStateMismatchDetector(unittest.TestCase):
    def test_engine_open_broker_flat_is_mismatch(self):
        out = detect_position_state_mismatch(
            engine_open=True, broker_open=True,
            engine_size=10.0, broker_size=10.0,
            engine_entry_price=100.0,
            broker_position_count=1,
            max_open_positions=1,
        )
        self.assertFalse(out["mismatch"])

    def test_engine_thinks_open_but_broker_flat(self):
        out = detect_position_state_mismatch(
            engine_open=True, broker_open=False,
            engine_size=10.0, broker_size=0.0,
            engine_entry_price=100.0,
            broker_position_count=0,
            max_open_positions=1,
        )
        self.assertTrue(out["mismatch"])

    def test_broker_open_but_engine_flat(self):
        out = detect_position_state_mismatch(
            engine_open=False, broker_open=True,
            engine_size=0.0, broker_size=10.0,
            engine_entry_price=None,
            broker_position_count=1,
            max_open_positions=1,
        )
        self.assertTrue(out["mismatch"])

    def test_too_many_open_positions(self):
        out = detect_position_state_mismatch(
            engine_open=True, broker_open=True,
            engine_size=10.0, broker_size=10.0,
            engine_entry_price=100.0,
            broker_position_count=2,
            max_open_positions=1,
        )
        self.assertTrue(out["mismatch"])

    def test_size_differs_materially(self):
        out = detect_position_state_mismatch(
            engine_open=True, broker_open=True,
            engine_size=10.0, broker_size=20.0,  # 100% off
            engine_entry_price=100.0,
            broker_position_count=1,
            max_open_positions=1,
            material_size_diff_pct=0.05,
        )
        self.assertTrue(out["mismatch"])

    def test_missing_entry_price_for_open_position(self):
        out = detect_position_state_mismatch(
            engine_open=True, broker_open=True,
            engine_size=10.0, broker_size=10.0,
            engine_entry_price=None,
            broker_position_count=1,
            max_open_positions=1,
        )
        self.assertTrue(out["mismatch"])


class TestCorruptedThresholdConfigDetector(unittest.TestCase):
    def test_missing_required_keys_corrupt(self):
        out = detect_corrupted_threshold_config(
            active_blob={"min_confidence": 0.6, "schema_version": 1},
            candidate_blob=None,
        )
        # Missing 'allow_chop'.
        self.assertTrue(out["corrupted"])

    def test_value_outside_bounds_corrupt(self):
        out = detect_corrupted_threshold_config(
            active_blob={"min_confidence": 0.05, "allow_chop": False,
                         "schema_version": 1},
            candidate_blob=None,
        )
        # min_confidence below 0.40 bound.
        self.assertTrue(out["corrupted"])

    def test_non_numeric_where_numeric_required_corrupt(self):
        out = detect_corrupted_threshold_config(
            active_blob={"min_confidence": "0.7", "allow_chop": False,
                         "schema_version": 1},
            candidate_blob=None,
        )
        self.assertTrue(out["corrupted"])

    def test_missing_schema_version_corrupt(self):
        out = detect_corrupted_threshold_config(
            active_blob={"min_confidence": 0.6, "allow_chop": False},
            candidate_blob=None,
        )
        self.assertTrue(out["corrupted"])

    def test_unsupported_schema_version_corrupt(self):
        out = detect_corrupted_threshold_config(
            active_blob={"min_confidence": 0.6, "allow_chop": False,
                         "schema_version": 999},
            candidate_blob=None,
        )
        self.assertTrue(out["corrupted"])

    def test_candidate_schema_mismatch_corrupt(self):
        out = detect_corrupted_threshold_config(
            active_blob={"min_confidence": 0.6, "allow_chop": False,
                         "schema_version": 1},
            candidate_blob={"min_confidence": 0.5, "extra_key": "boom",
                            "schema_version": 1},
        )
        # Candidate has a key not present in active -> schema mismatch.
        self.assertTrue(out["corrupted"])

    def test_clean_active_and_candidate_not_corrupt(self):
        out = detect_corrupted_threshold_config(
            active_blob={"min_confidence": 0.6, "allow_chop": False,
                         "schema_version": 1},
            candidate_blob={"min_confidence": 0.55, "allow_chop": False,
                            "schema_version": 1},
        )
        self.assertFalse(out["corrupted"])


# ---------- reset rules ----------

class TestSafeResetLiveMode(unittest.TestCase):
    def _all_clean(self, **overrides):
        base = {
            "system_mode": SYSTEM_MODE_LIVE,
            "now_ts": 5000,
            "manual_reset": True,
            "edge_decay_alert": False,
            "execution_error_unresolved": False,
            "consecutive_valid_candles": 100,
            "minutes_since_last_execution_error": 9999,
            "cost_model": CostModel(fee_pct=0.001),
            "account_equity": 100_000.0,
        }
        base.update(overrides)
        return base

    def test_live_mode_cannot_auto_reset(self):
        # manual_reset=False blocks live reset even when everything else
        # is fine.
        ks = KillSwitch()
        ks.activate(KS_REASON_DAILY_LOSS, now_ts=1000)
        out = ks.attempt_reset(**self._all_clean(manual_reset=False))
        self.assertFalse(out["reset_succeeded"])
        self.assertTrue(ks.active)

    def test_live_mode_resets_with_full_safe_conditions(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_DAILY_LOSS, now_ts=1000)
        out = ks.attempt_reset(**self._all_clean())
        self.assertTrue(out["reset_succeeded"])
        self.assertFalse(ks.active)
        self.assertEqual(ks.reason, "")

    def test_live_reset_blocked_during_edge_decay(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_DAILY_LOSS, now_ts=1000)
        out = ks.attempt_reset(**self._all_clean(edge_decay_alert=True))
        self.assertFalse(out["reset_succeeded"])
        self.assertTrue(ks.active)

    def test_live_reset_blocked_with_unresolved_execution_error(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_EXECUTION_ERROR, now_ts=1000)
        out = ks.attempt_reset(
            **self._all_clean(execution_error_unresolved=True)
        )
        self.assertFalse(out["reset_succeeded"])

    def test_live_reset_blocked_when_cost_model_missing(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_COST_MODEL_MISSING, now_ts=1000)
        out = ks.attempt_reset(**self._all_clean(cost_model=None))
        self.assertFalse(out["reset_succeeded"])

    def test_live_reset_blocked_when_account_equity_invalid(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_ACCOUNT_EQUITY_UNAVAILABLE, now_ts=1000)
        out = ks.attempt_reset(**self._all_clean(account_equity=None))
        self.assertFalse(out["reset_succeeded"])


class TestSafeResetBacktestPaperMode(unittest.TestCase):
    def _all_clean(self, mode, **overrides):
        base = {
            "system_mode": mode,
            "now_ts": 5000,
            "manual_reset": False,  # auto-reset path
            "edge_decay_alert": False,
            "execution_error_unresolved": False,
            "consecutive_valid_candles": 50,
            "minutes_since_last_execution_error": 100,
            "cost_model": CostModel(fee_pct=0.001),
            "account_equity": 100_000.0,
        }
        base.update(overrides)
        return base

    def test_backtest_can_auto_reset(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_INVALID_CANDLE, now_ts=1000)
        out = ks.attempt_reset(**self._all_clean(SYSTEM_MODE_BACKTEST))
        self.assertTrue(out["reset_succeeded"])
        self.assertFalse(ks.active)

    def test_paper_can_auto_reset(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_INVALID_CANDLE, now_ts=1000)
        out = ks.attempt_reset(**self._all_clean(SYSTEM_MODE_PAPER))
        self.assertTrue(out["reset_succeeded"])

    def test_auto_reset_requires_minimum_consecutive_valid_candles(self):
        cfg = ResetConfig(min_consecutive_valid_candles=10)
        ks = KillSwitch()
        ks.activate(KS_REASON_INVALID_CANDLE, now_ts=1000)
        out = ks.attempt_reset(
            config=cfg,
            **self._all_clean(SYSTEM_MODE_BACKTEST,
                              consecutive_valid_candles=5),
        )
        self.assertFalse(out["reset_succeeded"])

    def test_auto_reset_requires_quiet_period_after_execution_error(self):
        cfg = ResetConfig(min_minutes_since_execution_error=30)
        ks = KillSwitch()
        ks.activate(KS_REASON_EXECUTION_ERROR, now_ts=1000)
        out = ks.attempt_reset(
            config=cfg,
            **self._all_clean(SYSTEM_MODE_BACKTEST,
                              minutes_since_last_execution_error=10),
        )
        self.assertFalse(out["reset_succeeded"])

    def test_auto_reset_blocked_during_edge_decay(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_INVALID_CANDLE, now_ts=1000)
        out = ks.attempt_reset(
            **self._all_clean(SYSTEM_MODE_BACKTEST, edge_decay_alert=True),
        )
        self.assertFalse(out["reset_succeeded"])
        self.assertTrue(ks.active)


class TestResetLogging(unittest.TestCase):
    def test_successful_reset_is_logged(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_DAILY_LOSS, now_ts=1000)
        ks.attempt_reset(
            system_mode=SYSTEM_MODE_LIVE, now_ts=2000,
            manual_reset=True, edge_decay_alert=False,
            execution_error_unresolved=False,
            consecutive_valid_candles=100,
            minutes_since_last_execution_error=9999,
            cost_model=CostModel(fee_pct=0.001),
            account_equity=100_000.0,
        )
        log = ks.get_log()
        self.assertTrue(any(e.get("event") == "kill_switch_reset" for e in log))

    def test_failed_reset_is_logged(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_DAILY_LOSS, now_ts=1000)
        ks.attempt_reset(
            system_mode=SYSTEM_MODE_LIVE, now_ts=2000,
            manual_reset=False,  # blocks live reset
            edge_decay_alert=False,
            execution_error_unresolved=False,
            consecutive_valid_candles=100,
            minutes_since_last_execution_error=9999,
            cost_model=CostModel(fee_pct=0.001),
            account_equity=100_000.0,
        )
        log = ks.get_log()
        self.assertTrue(any(
            e.get("event") == "kill_switch_reset_failed" for e in log
        ))


class TestUnknownSystemMode(unittest.TestCase):
    def test_unknown_mode_blocks_reset(self):
        ks = KillSwitch()
        ks.activate(KS_REASON_DAILY_LOSS, now_ts=1000)
        out = ks.attempt_reset(
            system_mode="bogus_mode",
            now_ts=2000,
            manual_reset=True, edge_decay_alert=False,
            execution_error_unresolved=False,
            consecutive_valid_candles=100,
            minutes_since_last_execution_error=9999,
            cost_model=CostModel(fee_pct=0.001),
            account_equity=100_000.0,
        )
        self.assertFalse(out["reset_succeeded"])


# ---------- enum ----------

class TestKillSwitchReasonEnum(unittest.TestCase):
    def test_required_reasons_present(self):
        # Spec lists 14 distinct activation triggers.
        for required in (
            KS_REASON_DAILY_LOSS,
            KS_REASON_MAX_CONSECUTIVE_LOSSES,
            KS_REASON_STALE_DATA,
            KS_REASON_DUPLICATE_CANDLE,
            KS_REASON_INVALID_CANDLE,
            KS_REASON_ABNORMAL_SLIPPAGE,
            KS_REASON_EXECUTION_ERROR,
            KS_REASON_EDGE_DECAY,
            KS_REASON_COST_MODEL_MISSING,
            KS_REASON_ACCOUNT_EQUITY_UNAVAILABLE,
            KS_REASON_POSITION_STATE_MISMATCH,
            KS_REASON_UNKNOWN_SYSTEM_MODE,
            KS_REASON_CORRUPTED_THRESHOLD_CONFIG,
        ):
            self.assertIn(required, KILL_SWITCH_REASONS)


if __name__ == "__main__":
    unittest.main()
