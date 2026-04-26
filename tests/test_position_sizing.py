"""Tests for Prompt 2 / Step 5D - Position Sizing Safety."""

import unittest

from hermes.risk.sizing_safety import (
    REASON_EQUITY_UNAVAILABLE,
    REASON_INVALID_STOP_DISTANCE,
    REASON_OK,
    SizingSafetyConfig,
    safe_position_size,
)
from hermes.safety.data_contract import (
    SYSTEM_MODE_BACKTEST,
    SYSTEM_MODE_LIVE,
    SYSTEM_MODE_PAPER,
)


def _cfg(**overrides):
    base = {
        "max_risk_per_trade": 0.01,
        "max_position_pct_of_equity": 0.20,
        "absolute_risk_cap": 10_000.0,
        "confidence_multiplier_cap": 1.0,
    }
    base.update(overrides)
    return SizingSafetyConfig(**base)


# ------------------- output shape -------------------

class TestOutputShape(unittest.TestCase):
    def test_returns_canonical_keys(self):
        out = safe_position_size(
            equity=100_000.0, available_capital=100_000.0,
            confidence=0.5, stop_distance=2.0, config=_cfg(),
        )
        self.assertEqual(
            set(out.keys()),
            {"trade_allowed", "reason", "position_size", "details"},
        )
        self.assertIsInstance(out["trade_allowed"], bool)
        self.assertIsInstance(out["reason"], str)
        self.assertIsInstance(out["position_size"], float)
        self.assertIsInstance(out["details"], dict)


# ------------------- invalid stop distance -------------------

class TestInvalidStopDistance(unittest.TestCase):
    def test_zero_stop_distance_blocks_trade(self):
        out = safe_position_size(
            equity=100_000.0, available_capital=100_000.0,
            confidence=0.7, stop_distance=0.0, config=_cfg(),
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_STOP_DISTANCE)
        self.assertEqual(out["position_size"], 0.0)

    def test_negative_stop_distance_blocks_trade(self):
        out = safe_position_size(
            equity=100_000.0, available_capital=100_000.0,
            confidence=0.7, stop_distance=-0.1, config=_cfg(),
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_STOP_DISTANCE)


# ------------------- missing equity gating -------------------

class TestMissingEquity(unittest.TestCase):
    def test_missing_equity_blocks_paper_mode(self):
        out = safe_position_size(
            equity=None, available_capital=100_000.0,
            confidence=0.7, stop_distance=2.0,
            system_mode=SYSTEM_MODE_PAPER, config=_cfg(),
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_EQUITY_UNAVAILABLE)
        self.assertEqual(out["position_size"], 0.0)

    def test_missing_equity_blocks_live_mode(self):
        out = safe_position_size(
            equity=None, available_capital=100_000.0,
            confidence=0.7, stop_distance=2.0,
            system_mode=SYSTEM_MODE_LIVE, config=_cfg(),
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_EQUITY_UNAVAILABLE)

    def test_missing_equity_allowed_in_backtest(self):
        # Backtest may run without an equity reference; sizing falls back
        # to the absolute risk cap and any available_capital cap.
        out = safe_position_size(
            equity=None, available_capital=100_000.0,
            confidence=0.5, stop_distance=2.0,
            system_mode=SYSTEM_MODE_BACKTEST, config=_cfg(),
        )
        self.assertTrue(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_OK)
        self.assertGreater(out["position_size"], 0.0)

    def test_missing_equity_allowed_in_legacy_none_mode(self):
        # Legacy Prompt 1 callers without explicit mode are treated as
        # backtest for staleness purposes; the same applies to sizing.
        out = safe_position_size(
            equity=None, available_capital=100_000.0,
            confidence=0.5, stop_distance=2.0,
            system_mode=None, config=_cfg(),
        )
        self.assertTrue(out["trade_allowed"])


# ------------------- formula correctness -------------------

class TestFormulaCorrectness(unittest.TestCase):
    def test_risk_amount_uses_smaller_of_equity_pct_and_absolute_cap(self):
        # equity * max_risk_per_trade = 100_000 * 0.01 = 1_000.
        # absolute_risk_cap = 10_000. Smaller wins -> risk_amount = 1_000.
        cfg = _cfg(absolute_risk_cap=10_000.0)
        out = safe_position_size(
            equity=100_000.0, available_capital=1_000_000.0,
            confidence=1.0, stop_distance=2.0, config=cfg,
        )
        # risk_amount = 1_000 ; confidence_adjusted_risk = 1_000 * 1.0;
        # position_size = 1_000 / 2 = 500
        # equity_pct_cap = 100_000 * 0.20 = 20_000 (not binding)
        self.assertAlmostEqual(out["position_size"], 500.0, places=10)

    def test_absolute_cap_is_binding_when_smaller(self):
        cfg = _cfg(absolute_risk_cap=100.0)  # very tight
        out = safe_position_size(
            equity=100_000_000.0, available_capital=1e12,
            confidence=1.0, stop_distance=2.0, config=cfg,
        )
        # risk_amount = min(100_000_000*0.01=1_000_000, 100) = 100.
        # position_size = 100 / 2 = 50.
        self.assertAlmostEqual(out["position_size"], 50.0, places=10)

    def test_confidence_scales_position_size_linearly(self):
        cfg = _cfg()
        a = safe_position_size(100_000.0, 1e9, confidence=1.0,
                               stop_distance=2.0, config=cfg)["position_size"]
        b = safe_position_size(100_000.0, 1e9, confidence=0.5,
                               stop_distance=2.0, config=cfg)["position_size"]
        # Half confidence -> half size (other caps not binding).
        self.assertAlmostEqual(b, a * 0.5, places=10)

    def test_zero_confidence_zero_position_size(self):
        out = safe_position_size(
            100_000.0, 1e9, confidence=0.0, stop_distance=2.0, config=_cfg()
        )
        self.assertEqual(out["position_size"], 0.0)
        self.assertTrue(out["trade_allowed"])  # not blocked, just sized 0

    def test_stop_distance_inversely_scales_size(self):
        cfg = _cfg()
        a = safe_position_size(100_000.0, 1e9, 1.0, stop_distance=1.0,
                               config=cfg)["position_size"]
        b = safe_position_size(100_000.0, 1e9, 1.0, stop_distance=2.0,
                               config=cfg)["position_size"]
        self.assertAlmostEqual(b, a * 0.5, places=10)


# ------------------- caps -------------------

class TestCaps(unittest.TestCase):
    def test_position_never_exceeds_max_pct_of_equity(self):
        # Make risk-derived size huge, so equity-pct cap binds.
        cfg = _cfg(absolute_risk_cap=1e15, max_position_pct_of_equity=0.10)
        out = safe_position_size(
            equity=100.0, available_capital=1e15,
            confidence=1.0, stop_distance=0.001, config=cfg,
        )
        # equity_pct_cap = 100 * 0.10 = 10.
        self.assertLessEqual(out["position_size"], 10.0 + 1e-9)
        self.assertEqual(out["details"]["applied_cap"], "equity_pct")

    def test_position_never_exceeds_available_capital(self):
        cfg = _cfg(absolute_risk_cap=1e15, max_position_pct_of_equity=10.0)
        out = safe_position_size(
            equity=1e15, available_capital=42.0,
            confidence=1.0, stop_distance=0.001, config=cfg,
        )
        self.assertLessEqual(out["position_size"], 42.0 + 1e-9)
        self.assertEqual(out["details"]["applied_cap"], "available_capital")

    def test_high_confidence_does_not_bypass_caps(self):
        # Even with confidence_multiplier_cap pinned at 1.0 and confidence=1.0,
        # equity cap must bind when chosen.
        cfg = _cfg(
            absolute_risk_cap=1e15,
            max_position_pct_of_equity=0.05,
            confidence_multiplier_cap=1.0,
        )
        out = safe_position_size(
            equity=1_000.0, available_capital=1e15,
            confidence=1.0, stop_distance=0.001, config=cfg,
        )
        equity_pct_cap = 1_000.0 * 0.05
        self.assertLessEqual(out["position_size"], equity_pct_cap + 1e-9)

    def test_confidence_multiplier_cap_clamps_above_one(self):
        # Spec includes a confidence_multiplier_cap as a defensive cap.
        # Even if a hypothetical caller sneaks confidence >1, it must be
        # clamped to the cap. (We reject confidence>1 outright via
        # is_unit_interval; assert the cap path is reachable when
        # cap < 1.0.)
        cfg = _cfg(confidence_multiplier_cap=0.5,
                   absolute_risk_cap=1_000_000.0)
        out_capped = safe_position_size(
            100_000.0, 1e9, confidence=1.0, stop_distance=2.0, config=cfg
        )["position_size"]
        out_uncapped = safe_position_size(
            100_000.0, 1e9, confidence=0.5, stop_distance=2.0, config=cfg
        )["position_size"]
        self.assertAlmostEqual(out_capped, out_uncapped, places=10)


# ------------------- determinism + input validation -------------------

class TestDeterminismAndInputs(unittest.TestCase):
    def test_same_input_same_output(self):
        cfg = _cfg()
        a = safe_position_size(100_000.0, 1e9, 0.7, 2.0, config=cfg)
        b = safe_position_size(100_000.0, 1e9, 0.7, 2.0, config=cfg)
        self.assertEqual(a, b)

    def test_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            safe_position_size(100_000.0, 1e9, confidence=1.5,
                               stop_distance=2.0, config=_cfg())
        with self.assertRaises(ValueError):
            safe_position_size(100_000.0, 1e9, confidence=-0.1,
                               stop_distance=2.0, config=_cfg())

    def test_rejects_negative_equity(self):
        with self.assertRaises(ValueError):
            safe_position_size(-1.0, 1e9, 0.5, 2.0, config=_cfg())

    def test_rejects_negative_available_capital(self):
        with self.assertRaises(ValueError):
            safe_position_size(100_000.0, -1.0, 0.5, 2.0, config=_cfg())


# ------------------- config -------------------

class TestSizingSafetyConfig(unittest.TestCase):
    def test_defaults_present_and_reasonable(self):
        c = SizingSafetyConfig()
        self.assertGreater(c.max_risk_per_trade, 0.0)
        self.assertLessEqual(c.max_risk_per_trade, 1.0)
        self.assertGreater(c.max_position_pct_of_equity, 0.0)
        self.assertGreater(c.absolute_risk_cap, 0.0)
        self.assertGreater(c.confidence_multiplier_cap, 0.0)

    def test_rejects_negative_max_risk_per_trade(self):
        with self.assertRaises(ValueError):
            SizingSafetyConfig(max_risk_per_trade=-0.01)

    def test_rejects_max_risk_per_trade_above_one(self):
        with self.assertRaises(ValueError):
            SizingSafetyConfig(max_risk_per_trade=1.5)

    def test_rejects_negative_absolute_risk_cap(self):
        with self.assertRaises(ValueError):
            SizingSafetyConfig(absolute_risk_cap=-1.0)

    def test_rejects_zero_confidence_multiplier_cap(self):
        with self.assertRaises(ValueError):
            SizingSafetyConfig(confidence_multiplier_cap=0.0)


if __name__ == "__main__":
    unittest.main()
