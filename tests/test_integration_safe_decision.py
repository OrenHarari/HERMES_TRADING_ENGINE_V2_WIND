"""Tests for the Integration Layer - safe_make_decision.

These tests cover ONLY the wrapper. Phase-1 unit tests (268) and Phase-2
unit tests (293) live in their own files and are not duplicated here.
"""

import json
import unittest

from hermes.decision.decision import make_decision
from hermes.integration import (
    INTEGRATION_OUTPUT_KEYS,
    PHASE2_ADDITIVE_KEYS,
    STAGE_CORE,
    STAGE_COST_MODEL,
    STAGE_DATA_CONTRACT,
    STAGE_KILL_SWITCH,
    STAGE_LIVE_GATE,
    STAGE_NONE,
    STAGE_SIZING_SAFETY,
    safe_make_decision,
)
from hermes.risk.sizing_safety import SizingSafetyConfig
from hermes.safety.cost_model import CostModel
from hermes.safety.data_contract import (
    SYSTEM_MODE_BACKTEST,
    SYSTEM_MODE_LIVE,
    SYSTEM_MODE_PAPER,
)
from hermes.safety.kill_switch import KS_REASON_DAILY_LOSS, KillSwitch


# ---- helpers --------------------------------------------------------------

def _candles_uptrend(n=25, start_price=100.0, step=0.1, start_ts=1000, ts_step=60):
    out = []
    p = start_price
    for i in range(n):
        out.append(
            {
                "timestamp": start_ts + i * ts_step,
                "open": p,
                "high": p + 0.001,
                "low": p - 0.001,
                "close": p,
            }
        )
        p += step
    return out


def _candles_flat(n=25, p=100.0, start_ts=1000, ts_step=60):
    return [
        {
            "timestamp": start_ts + i * ts_step,
            "open": p, "high": p + 0.001, "low": p - 0.001, "close": p,
        }
        for i in range(n)
    ]


def _high_signal():
    return {"sequence_value": 0.85, "amd_value": 0.85, "combined_value": 0.85}


def _low_signal():
    return {"sequence_value": 0.10, "amd_value": 0.10, "combined_value": 0.10}


def _passing_paper_validation():
    return {"paper_validation_passed": True, "live_enabled": False,
            "reason": "", "details": "",
            "insufficient_regime_diversity": False}


def _failing_paper_validation():
    return {"paper_validation_passed": False, "live_enabled": False,
            "reason": "paper_validation_failed",
            "details": "insufficient_paper_trades",
            "insufficient_regime_diversity": False}


# =========================================================================
# Output shape
# =========================================================================
class TestOutputShape(unittest.TestCase):
    def test_approved_output_has_canonical_keyset(self):
        out = safe_make_decision(_high_signal(), _candles_uptrend(), 24)
        self.assertEqual(set(out.keys()), set(INTEGRATION_OUTPUT_KEYS))

    def test_blocked_output_has_canonical_keyset_data_contract(self):
        # Bad candle data -> blocked at data_contract stage.
        bad = _candles_uptrend()
        bad[3]["high"] = -1.0  # OHLC violation
        bad[3]["low"] = -2.0
        out = safe_make_decision(_high_signal(), bad, 24)
        self.assertEqual(set(out.keys()), set(INTEGRATION_OUTPUT_KEYS))
        self.assertFalse(out["trade_allowed"])

    def test_output_is_json_serializable(self):
        out = safe_make_decision(_high_signal(), _candles_uptrend(), 24)
        # Round-trip via JSON to confirm.
        json.dumps(out)

    def test_position_size_never_negative(self):
        # Force a block path; position_size must be 0.0, never negative.
        out = safe_make_decision(
            _high_signal(),
            _candles_uptrend(),
            24,
            system_mode="bogus_mode",
        )
        self.assertGreaterEqual(out["position_size"], 0.0)

    def test_phase1_keys_subset_of_output(self):
        out = safe_make_decision(_high_signal(), _candles_uptrend(), 24)
        for k in (
            "trade_allowed", "confidence", "agreement", "regime",
            "position_size", "reason",
        ):
            self.assertIn(k, out)


# =========================================================================
# Stage precedence (first failing stage wins)
# =========================================================================
class TestStagePrecedence(unittest.TestCase):
    def test_invalid_mode_beats_everything(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode="bogus_mode",
            kill_switch=_active_kill_switch(),
            cost_model=None,
        )
        self.assertEqual(out["reason"], "invalid_system_mode")
        self.assertEqual(out["blocked_by_stage"], STAGE_DATA_CONTRACT)

    def test_invalid_data_beats_kill_switch_and_cost(self):
        ks = _active_kill_switch()
        bad = _candles_uptrend()
        bad[10]["high"] = 0.0
        bad[10]["low"] = 100.0
        out = safe_make_decision(
            _high_signal(), bad, 24,
            system_mode=SYSTEM_MODE_PAPER, kill_switch=ks,
            cost_model=None, now_ts=2000,
        )
        self.assertEqual(out["reason"], "invalid_market_data")
        self.assertEqual(out["blocked_by_stage"], STAGE_DATA_CONTRACT)

    def test_kill_switch_beats_cost_and_paper_gate(self):
        ks = _active_kill_switch()
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_LIVE,
            kill_switch=ks,
            cost_model=None,  # would otherwise also block
            live_enabled=False,
            paper_validation_result=_failing_paper_validation(),
        )
        self.assertEqual(out["reason"], "kill_switch_active")
        self.assertEqual(out["blocked_by_stage"], STAGE_KILL_SWITCH)
        self.assertTrue(out["kill_switch_active"])
        self.assertEqual(out["kill_switch_reason"], KS_REASON_DAILY_LOSS)

    def test_missing_cost_model_in_live_beats_paper_gate(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_LIVE,
            cost_model=None,
            live_enabled=False,  # would otherwise also block
            paper_validation_result=_failing_paper_validation(),
        )
        self.assertEqual(out["reason"], "missing_cost_model")
        self.assertEqual(out["blocked_by_stage"], STAGE_COST_MODEL)

    def test_paper_gate_blocks_when_live_not_enabled(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_LIVE,
            cost_model=CostModel(),
            live_enabled=False,
            paper_validation_result=_passing_paper_validation(),
        )
        self.assertEqual(out["reason"], "live_not_explicitly_enabled")
        self.assertEqual(out["blocked_by_stage"], STAGE_LIVE_GATE)

    def test_paper_gate_blocks_when_paper_validation_failed(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_LIVE,
            cost_model=CostModel(),
            live_enabled=True,
            paper_validation_result=_failing_paper_validation(),
        )
        self.assertEqual(out["reason"], "paper_validation_failed")
        self.assertEqual(out["blocked_by_stage"], STAGE_LIVE_GATE)

    def test_core_block_low_confidence(self):
        out = safe_make_decision(_low_signal(), _candles_flat(), 24)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["blocked_by_stage"], STAGE_CORE)
        # Phase-1 reason preserved.
        self.assertIn(
            out["reason"],
            {"low_confidence", "low_agreement", "volatility_too_low",
             "regime_chop_disallowed"},
        )

    def test_core_passes_then_sizing_safety_blocks_invalid_stop(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_PAPER,
            cost_model=CostModel(),
            account_equity=100_000.0,
            stop_distance=0.0,  # invalid
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "invalid_stop_distance")
        self.assertEqual(out["blocked_by_stage"], STAGE_SIZING_SAFETY)

    def test_core_passes_then_sizing_safety_blocks_no_equity(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_PAPER,
            cost_model=CostModel(),
            account_equity=None,  # blocks in paper
            stop_distance=2.0,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "account_equity_unavailable")
        self.assertEqual(out["blocked_by_stage"], STAGE_SIZING_SAFETY)

    def test_blocked_by_stage_is_empty_string_when_approved(self):
        out = safe_make_decision(_high_signal(), _candles_uptrend(), 24)
        self.assertTrue(out["trade_allowed"])
        self.assertEqual(out["blocked_by_stage"], STAGE_NONE)


def _active_kill_switch():
    ks = KillSwitch()
    ks.activate(KS_REASON_DAILY_LOSS, now_ts=0)
    return ks


# =========================================================================
# Mode-conditional behavior
# =========================================================================
class TestModeConditional(unittest.TestCase):
    def test_legacy_none_mode_no_safety_gates(self):
        # No staleness, no cost-model gate, no live-gate, no safe sizing.
        # Behaves identically to make_decision in shape (plus additive keys).
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=None, now_ts=10**12,
            max_staleness_seconds=10,
        )
        self.assertTrue(out["trade_allowed"])

    def test_backtest_mode_no_safety_gates(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_BACKTEST,
            now_ts=10**12, max_staleness_seconds=10,
        )
        self.assertTrue(out["trade_allowed"])

    def test_paper_mode_passes_with_full_inputs(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_PAPER,
            cost_model=CostModel(),
            account_equity=100_000.0,
            stop_distance=2.0,
        )
        self.assertTrue(out["trade_allowed"])
        self.assertEqual(out["system_mode"], "paper_mode")

    def test_live_mode_blocked_when_live_enabled_false(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_LIVE,
            cost_model=CostModel(),
            account_equity=100_000.0,
            stop_distance=2.0,
            live_enabled=False,
            paper_validation_result=_passing_paper_validation(),
        )
        self.assertEqual(out["reason"], "live_not_explicitly_enabled")

    def test_live_mode_blocked_when_no_paper_validation(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_LIVE,
            cost_model=CostModel(),
            account_equity=100_000.0,
            stop_distance=2.0,
            live_enabled=True,
            paper_validation_result=None,
        )
        self.assertEqual(out["reason"], "paper_validation_failed")

    def test_live_mode_passes_with_full_inputs(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_LIVE,
            cost_model=CostModel(),
            account_equity=100_000.0,
            stop_distance=2.0,
            live_enabled=True,
            paper_validation_result=_passing_paper_validation(),
        )
        self.assertTrue(out["trade_allowed"])
        self.assertTrue(out["live_enabled"])
        self.assertTrue(out["paper_validation_passed"])
        self.assertTrue(out["cost_model_applied"])

    def test_paper_mode_blocks_stale_data(self):
        # last visible candle ts = 1000 + 24*60 = 2440
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_PAPER,
            cost_model=CostModel(),
            account_equity=100_000.0, stop_distance=2.0,
            now_ts=10**6,  # very far in the future
            max_staleness_seconds=120,
        )
        self.assertEqual(out["reason"], "stale_market_data")
        self.assertEqual(out["blocked_by_stage"], STAGE_DATA_CONTRACT)

    def test_backtest_mode_does_not_block_stale_data(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_BACKTEST,
            now_ts=10**6, max_staleness_seconds=120,
        )
        self.assertTrue(out["trade_allowed"])


# =========================================================================
# Confidence cannot bypass safety
# =========================================================================
class TestConfidenceCannotBypass(unittest.TestCase):
    def test_kill_switch_blocks_even_with_max_signal(self):
        ks = _active_kill_switch()
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            kill_switch=ks,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "kill_switch_active")

    def test_live_gate_blocks_even_with_max_signal(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_LIVE,
            cost_model=CostModel(),
            live_enabled=False,
            paper_validation_result=_passing_paper_validation(),
            account_equity=100_000.0, stop_distance=2.0,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "live_not_explicitly_enabled")

    def test_cost_model_missing_blocks_even_with_max_signal(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_LIVE,
            cost_model=None,
            live_enabled=True,
            paper_validation_result=_passing_paper_validation(),
            account_equity=100_000.0, stop_distance=2.0,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "missing_cost_model")


# =========================================================================
# Sizing wrap behavior
# =========================================================================
class TestSizingWrap(unittest.TestCase):
    def test_paper_mode_overlay_can_reduce_size(self):
        # A very tight equity-pct cap forces safe sizing to clamp.
        cfg = SizingSafetyConfig(
            max_position_pct_of_equity=0.0001,  # 0.01% of equity
            absolute_risk_cap=1.0,
        )
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_PAPER,
            cost_model=CostModel(),
            account_equity=100.0, stop_distance=2.0,
            safe_sizing_config=cfg,
        )
        self.assertTrue(out["trade_allowed"])
        self.assertLessEqual(out["position_size"], 100.0 * 0.0001 + 1e-9)

    def test_paper_overlay_never_increases_size(self):
        # Compare wrapper paper-mode sizing against core sizing.
        core = make_decision(_high_signal(), _candles_uptrend(), 24)
        wrap = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_PAPER,
            cost_model=CostModel(),
            account_equity=1_000_000.0, stop_distance=2.0,
        )
        self.assertTrue(wrap["trade_allowed"])
        self.assertLessEqual(wrap["position_size"], core["position_size"] + 1e-9)

    def test_backtest_mode_size_equals_core_size(self):
        core = make_decision(_high_signal(), _candles_uptrend(), 24)
        wrap = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_BACKTEST,
        )
        self.assertEqual(wrap["position_size"], core["position_size"])

    def test_legacy_none_mode_size_equals_core_size(self):
        core = make_decision(_high_signal(), _candles_uptrend(), 24)
        wrap = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=None,
        )
        self.assertEqual(wrap["position_size"], core["position_size"])


# =========================================================================
# Determinism + no future data
# =========================================================================
class TestDeterminismAndNoFutureData(unittest.TestCase):
    def test_same_inputs_same_output(self):
        a = safe_make_decision(_high_signal(), _candles_uptrend(), 24)
        b = safe_make_decision(_high_signal(), _candles_uptrend(), 24)
        self.assertEqual(a, b)

    def test_no_future_data_used(self):
        candles = _candles_uptrend(30)
        before = safe_make_decision(_high_signal(), candles, 14)
        for j in range(15, len(candles)):
            candles[j] = {
                "timestamp": 10**9 + j,
                "open": 1e9, "high": 1e9, "low": 1e9, "close": 1e9,
            }
        after = safe_make_decision(_high_signal(), candles, 14)
        self.assertEqual(before, after)

    def test_kill_switch_state_unchanged_on_pass(self):
        ks = KillSwitch()  # inactive
        before = ks.state()
        safe_make_decision(
            _high_signal(), _candles_uptrend(), 24, kill_switch=ks,
        )
        after = ks.state()
        self.assertEqual(before, after)


# =========================================================================
# Backward-compat regression
# =========================================================================
class TestBackwardCompatRegression(unittest.TestCase):
    def test_make_decision_directly_still_returns_six_keys(self):
        out = make_decision(_high_signal(), _candles_uptrend(), 24)
        self.assertEqual(
            set(out.keys()),
            {"trade_allowed", "confidence", "agreement", "regime",
             "position_size", "reason"},
        )

    def test_wrapper_phase1_keys_match_phase1_types(self):
        out = safe_make_decision(_high_signal(), _candles_uptrend(), 24)
        self.assertIsInstance(out["trade_allowed"], bool)
        self.assertIsInstance(out["confidence"], float)
        self.assertIsInstance(out["agreement"], float)
        self.assertIsInstance(out["regime"], str)
        self.assertIsInstance(out["position_size"], float)
        self.assertIsInstance(out["reason"], str)

    def test_wrapper_additive_keys_have_expected_types(self):
        out = safe_make_decision(_high_signal(), _candles_uptrend(), 24)
        self.assertIsInstance(out["system_mode"], str)
        self.assertIsInstance(out["kill_switch_active"], bool)
        self.assertIsInstance(out["kill_switch_reason"], str)
        self.assertIsInstance(out["cost_model_applied"], bool)
        self.assertIsInstance(out["live_enabled"], bool)
        self.assertIsInstance(out["paper_validation_passed"], bool)
        self.assertIsInstance(out["blocked_by_stage"], str)


# =========================================================================
# Integration with existing safety state objects
# =========================================================================
class TestSafetyObjectIntegration(unittest.TestCase):
    def test_kill_switch_state_read_not_mutated(self):
        ks = KillSwitch()  # inactive
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24, kill_switch=ks,
        )
        self.assertFalse(out["kill_switch_active"])
        self.assertFalse(ks.active)

    def test_cost_model_treated_as_presence_gate_only(self):
        # The wrapper does NOT compute pnl/cost math.
        cm = CostModel(fee_pct=0.5, slippage_pct=0.5)  # absurd costs
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_PAPER,
            cost_model=cm,
            account_equity=100_000.0, stop_distance=2.0,
        )
        self.assertTrue(out["trade_allowed"])
        self.assertTrue(out["cost_model_applied"])

    def test_paper_validation_result_dict_consumed(self):
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_LIVE,
            cost_model=CostModel(),
            account_equity=100_000.0, stop_distance=2.0,
            live_enabled=True,
            paper_validation_result=_passing_paper_validation(),
        )
        self.assertTrue(out["paper_validation_passed"])

    def test_safe_sizing_config_forwarded(self):
        cfg = SizingSafetyConfig(absolute_risk_cap=0.0)  # forces zero size
        out = safe_make_decision(
            _high_signal(), _candles_uptrend(), 24,
            system_mode=SYSTEM_MODE_PAPER,
            cost_model=CostModel(),
            account_equity=100_000.0, stop_distance=2.0,
            safe_sizing_config=cfg,
        )
        self.assertEqual(out["position_size"], 0.0)
        # Still trade_allowed True; sizing simply returns 0.
        self.assertTrue(out["trade_allowed"])

    def test_invalid_kill_switch_type_rejected(self):
        with self.assertRaises(ValueError):
            safe_make_decision(
                _high_signal(), _candles_uptrend(), 24,
                kill_switch="not a KillSwitch",
            )

    def test_invalid_cost_model_type_rejected(self):
        with self.assertRaises(ValueError):
            safe_make_decision(
                _high_signal(), _candles_uptrend(), 24,
                cost_model={"fee_pct": 0.001},
            )

    def test_invalid_live_enabled_type_rejected(self):
        with self.assertRaises(ValueError):
            safe_make_decision(
                _high_signal(), _candles_uptrend(), 24,
                live_enabled="yes",
            )


# =========================================================================
# Re-export sanity
# =========================================================================
class TestPublicSurface(unittest.TestCase):
    def test_safe_make_decision_re_exported_from_hermes(self):
        # Import via the top-level package re-export.
        from hermes import safe_make_decision as top_level
        self.assertIs(top_level, safe_make_decision)

    def test_phase2_additive_keys_exact_set(self):
        self.assertEqual(
            set(PHASE2_ADDITIVE_KEYS),
            {
                "system_mode",
                "kill_switch_active",
                "kill_switch_reason",
                "cost_model_applied",
                "live_enabled",
                "paper_validation_passed",
                "blocked_by_stage",
            },
        )


if __name__ == "__main__":
    unittest.main()
