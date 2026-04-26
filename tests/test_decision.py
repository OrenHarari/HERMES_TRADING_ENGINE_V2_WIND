"""Tests for Prompt 1 / Step 5 - Top-level make_decision()."""

import unittest

from hermes.decision import (
    DecisionConfig,
    REQUIRED_DECISION_KEYS,
    make_decision,
)
from hermes.market import MarketIntelligenceConfig
from hermes.risk import RiskConfig, RiskState


def _candles_uptrend(n=25, start=100.0, step=0.1):
    out = []
    p = start
    for _ in range(n):
        out.append(
            {
                "open": p,
                "high": p + 0.001,
                "low": p - 0.001,
                "close": p,
            }
        )
        p += step
    return out


def _candles_flat(n=25, p=100.0):
    return [{"open": p, "high": p, "low": p, "close": p} for _ in range(n)]


def _high_signal():
    return {"sequence_value": 0.85, "amd_value": 0.85, "combined_value": 0.85}


def _low_signal():
    return {"sequence_value": 0.10, "amd_value": 0.10, "combined_value": 0.10}


class TestMakeDecisionContract(unittest.TestCase):
    def test_required_keys_match_spec(self):
        self.assertEqual(
            set(REQUIRED_DECISION_KEYS),
            {"trade_allowed", "confidence", "agreement", "regime",
             "position_size", "reason"},
        )

    def test_returns_exact_key_set_when_allowed(self):
        out = make_decision(_high_signal(), _candles_uptrend(), 24)
        self.assertEqual(set(out.keys()), set(REQUIRED_DECISION_KEYS))

    def test_returns_exact_key_set_when_blocked(self):
        out = make_decision(_low_signal(), _candles_flat(), 24)
        self.assertEqual(set(out.keys()), set(REQUIRED_DECISION_KEYS))

    def test_value_types(self):
        out = make_decision(_high_signal(), _candles_uptrend(), 24)
        self.assertIsInstance(out["trade_allowed"], bool)
        self.assertIsInstance(out["confidence"], float)
        self.assertIsInstance(out["agreement"], float)
        self.assertIsInstance(out["regime"], str)
        self.assertIsInstance(out["position_size"], float)
        self.assertIsInstance(out["reason"], str)


class TestMakeDecisionAllowance(unittest.TestCase):
    def test_high_quality_signal_in_uptrend_is_allowed(self):
        out = make_decision(_high_signal(), _candles_uptrend(), 24)
        self.assertTrue(out["trade_allowed"], msg="reason: " + out["reason"])
        self.assertEqual(out["reason"], "")
        self.assertGreater(out["position_size"], 0.0)

    def test_low_signal_blocked_for_low_confidence(self):
        out = make_decision(_low_signal(), _candles_flat(), 24)
        self.assertFalse(out["trade_allowed"])
        # Flat -> low_volatility regime; confidence will be very low.
        # Blocking reason should be one of the eligibility reasons.
        self.assertIn(
            out["reason"],
            {"low_confidence", "low_agreement", "volatility_too_low",
             "regime_chop_disallowed"},
        )

    def test_position_size_zero_when_blocked(self):
        out = make_decision(_low_signal(), _candles_flat(), 24)
        self.assertEqual(out["position_size"], 0.0)


class TestMakeDecisionRiskPrecedence(unittest.TestCase):
    def test_risk_block_takes_precedence_over_high_confidence(self):
        # Pre-fill risk state with too many losses so it's blocked.
        rs = RiskState(RiskConfig(max_consecutive_losses=2, cooldown_seconds=0))
        rs.register_trade_outcome(-10.0, "d1")
        rs.register_trade_outcome(-10.0, "d1")
        out = make_decision(
            _high_signal(),
            _candles_uptrend(),
            24,
            risk_state=rs,
            now_ts=1000,
            day_key="d1",
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "max_consecutive_losses_reached")
        self.assertEqual(out["position_size"], 0.0)

    def test_high_confidence_does_not_bypass_max_trades_per_day(self):
        rs = RiskState(RiskConfig(max_trades_per_day=1, cooldown_seconds=0))
        rs.register_trade_entry(0, "d1")
        out = make_decision(
            _high_signal(),
            _candles_uptrend(),
            24,
            risk_state=rs,
            now_ts=10,
            day_key="d1",
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "max_trades_per_day_reached")

    def test_daily_loss_limit_blocks_even_high_confidence(self):
        rs = RiskState(RiskConfig(max_daily_loss=20.0, cooldown_seconds=0))
        rs.register_trade_outcome(-25.0, "d1")
        out = make_decision(
            _high_signal(),
            _candles_uptrend(),
            24,
            risk_state=rs,
            now_ts=0,
            day_key="d1",
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "max_daily_loss_reached")


class TestMakeDecisionDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        a = make_decision(_high_signal(), _candles_uptrend(), 24)
        b = make_decision(_high_signal(), _candles_uptrend(), 24)
        self.assertEqual(a, b)

    def test_no_future_data_used(self):
        candles = _candles_uptrend(30)
        idx = 14
        before = make_decision(_high_signal(), candles, idx)
        for j in range(idx + 1, len(candles)):
            candles[j] = {"open": 1e9, "high": 1e9, "low": 1e9, "close": 1e9}
        after = make_decision(_high_signal(), candles, idx)
        self.assertEqual(before, after)


class TestMakeDecisionPhase2Hooks(unittest.TestCase):
    def test_system_mode_kwarg_accepted_and_ignored(self):
        # Phase 1 must not act on these args; output must be identical.
        a = make_decision(_high_signal(), _candles_uptrend(), 24)
        b = make_decision(
            _high_signal(),
            _candles_uptrend(),
            24,
            system_mode="paper",
            safety_context={"foo": "bar"},
        )
        self.assertEqual(a, b)


class TestMakeDecisionConfigOverrides(unittest.TestCase):
    def test_min_confidence_override_can_block(self):
        cfg = DecisionConfig(min_confidence=0.99)
        out = make_decision(
            _high_signal(),
            _candles_uptrend(),
            24,
            decision_config=cfg,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "low_confidence")

    def test_market_config_override_changes_intelligence(self):
        # A tiny lookback can shift the regime; just verify the call works.
        mc = MarketIntelligenceConfig(lookback=5)
        out = make_decision(
            _high_signal(),
            _candles_uptrend(),
            24,
            market_config=mc,
        )
        self.assertIn(out["regime"], {
            "trend_up", "trend_down", "chop", "high_volatility", "low_volatility"
        })


if __name__ == "__main__":
    unittest.main()
