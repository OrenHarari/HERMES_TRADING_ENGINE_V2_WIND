"""Tests for Prompt 2 / Step 6D - Out-of-Sample Promotion Gate."""

import os
import shutil
import tempfile
import unittest

from hermes.learning.candidate_thresholds import (
    REASON_CANDIDATE_REJECTED_OOS,
    ThresholdStore,
    promote_candidate,
    propose_candidate,
)
from hermes.learning.oos_gate import (
    OOSValidationConfig,
    REASON_INSUFFICIENT_VALIDATION_TRADES,
    REASON_MAX_DRAWDOWN_INCREASED,
    REASON_PROFIT_FACTOR_DETERIORATED,
    REASON_STABILITY_TOO_LOW,
    REASON_TRADE_COUNT_REDUCED,
    REASON_VALIDATION_NOT_AFTER_DISCOVERY,
    REASON_WIN_RATE_BUT_RISK_WORSENS,
    evaluate_oos_promotion,
    split_trades_into_oos_windows,
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
        "avg_win": 10.0,
        "avg_loss": -5.0,
        "trades_per_regime": {},
        "cost_model_applied": True,
    }


def _windows():
    return ("2025-01", "2025-06"), ("2025-07", "2025-08")


def _trade(date, regime="trend_up", pnl=1.0):
    return {"date": date, "regime": regime, "pnl": pnl, "net_pnl": pnl}


# ---------- window structure ----------

class TestWindowStructure(unittest.TestCase):
    def test_validation_must_be_strictly_later(self):
        # Valid windows pass.
        out = evaluate_oos_promotion(
            discovery_window=("2025-01", "2025-06"),
            validation_window=("2025-07", "2025-08"),
            active_validation_report=_report(),
            candidate_validation_report=_report(),
        )
        self.assertTrue(out["validation_passed"])

    def test_overlapping_windows_rejected(self):
        out = evaluate_oos_promotion(
            discovery_window=("2025-01", "2025-06"),
            validation_window=("2025-06", "2025-08"),
            active_validation_report=_report(),
            candidate_validation_report=_report(),
        )
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], REASON_VALIDATION_NOT_AFTER_DISCOVERY)

    def test_validation_earlier_than_discovery_rejected(self):
        out = evaluate_oos_promotion(
            discovery_window=("2025-07", "2025-08"),
            validation_window=("2025-01", "2025-06"),
            active_validation_report=_report(),
            candidate_validation_report=_report(),
        )
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], REASON_VALIDATION_NOT_AFTER_DISCOVERY)


# ---------- validation trade count ----------

class TestValidationTradeCount(unittest.TestCase):
    def test_below_minimum_rejected(self):
        cfg = OOSValidationConfig(min_validation_trades=30)
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(trade_count=10),
            candidate_validation_report=_report(trade_count=10),
            config=cfg,
        )
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], REASON_INSUFFICIENT_VALIDATION_TRADES)

    def test_at_minimum_accepted(self):
        cfg = OOSValidationConfig(min_validation_trades=30)
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(trade_count=30),
            candidate_validation_report=_report(trade_count=30),
            config=cfg,
        )
        self.assertTrue(out["validation_passed"])

    def test_candidate_reduces_trade_count_too_far(self):
        cfg = OOSValidationConfig(
            min_validation_trades=10,
            min_trade_count_ratio=0.5,
        )
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(trade_count=100),
            candidate_validation_report=_report(trade_count=20),  # 0.20 ratio
            config=cfg,
        )
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], REASON_TRADE_COUNT_REDUCED)


# ---------- material deterioration of metrics ----------

class TestMetricDeterioration(unittest.TestCase):
    def test_profit_factor_deterioration_rejected(self):
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(profit_factor=2.0),
            candidate_validation_report=_report(profit_factor=1.0),
        )
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], REASON_PROFIT_FACTOR_DETERIORATED)

    def test_profit_factor_within_tolerance_accepted(self):
        cfg = OOSValidationConfig(profit_factor_min_ratio=0.95)
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(profit_factor=1.5),
            # Slight dip but within 95% tolerance.
            candidate_validation_report=_report(profit_factor=1.45),
            config=cfg,
        )
        self.assertTrue(out["validation_passed"])

    def test_max_drawdown_material_increase_rejected(self):
        cfg = OOSValidationConfig(max_drawdown_max_increase_pct=0.20)
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(max_drawdown=100.0),
            candidate_validation_report=_report(max_drawdown=200.0),
            config=cfg,
        )
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], REASON_MAX_DRAWDOWN_INCREASED)

    def test_max_drawdown_within_tolerance_accepted(self):
        cfg = OOSValidationConfig(max_drawdown_max_increase_pct=0.20)
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(max_drawdown=100.0),
            candidate_validation_report=_report(max_drawdown=110.0),
            config=cfg,
        )
        self.assertTrue(out["validation_passed"])

    def test_stability_below_minimum_rejected(self):
        cfg = OOSValidationConfig(min_stability_score=0.30)
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(stability_score=0.50),
            candidate_validation_report=_report(stability_score=0.10),
            config=cfg,
        )
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], REASON_STABILITY_TOO_LOW)


# ---------- "win_rate alone is not enough" ----------

class TestWinRateAloneNotEnough(unittest.TestCase):
    def test_win_rate_improves_but_drawdown_worsens_materially(self):
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(win_rate=0.50, max_drawdown=100.0,
                                             profit_factor=1.5),
            # Higher win rate but drawdown 4x worse.
            candidate_validation_report=_report(win_rate=0.70, max_drawdown=400.0,
                                                profit_factor=1.5),
        )
        self.assertFalse(out["validation_passed"])
        self.assertIn(out["reason"], (
            REASON_MAX_DRAWDOWN_INCREASED,
            REASON_WIN_RATE_BUT_RISK_WORSENS,
        ))

    def test_win_rate_improves_but_profit_factor_deteriorates(self):
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(win_rate=0.50, profit_factor=2.0),
            candidate_validation_report=_report(win_rate=0.65, profit_factor=1.0),
        )
        self.assertFalse(out["validation_passed"])


# ---------- strong OOS allows promotion ----------

class TestStrongOOSAllowsPromotion(unittest.TestCase):
    def test_strong_candidate_passes(self):
        d, v = _windows()
        out = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(profit_factor=1.5,
                                             max_drawdown=100.0,
                                             stability_score=0.5,
                                             win_rate=0.55,
                                             trade_count=100),
            candidate_validation_report=_report(profit_factor=1.8,
                                                max_drawdown=80.0,
                                                stability_score=0.6,
                                                win_rate=0.60,
                                                trade_count=95),
        )
        self.assertTrue(out["validation_passed"])
        self.assertEqual(out["reason"], "")


# ---------- determinism ----------

class TestDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        d, v = _windows()
        a = evaluate_oos_promotion(d, v, _report(), _report())
        b = evaluate_oos_promotion(d, v, _report(), _report())
        self.assertEqual(a, b)


# ---------- output shape ----------

class TestOutputShape(unittest.TestCase):
    def test_canonical_keys(self):
        d, v = _windows()
        out = evaluate_oos_promotion(d, v, _report(), _report())
        self.assertEqual(
            set(out.keys()), {"validation_passed", "reason", "details"}
        )
        self.assertIsInstance(out["validation_passed"], bool)
        self.assertIsInstance(out["reason"], str)
        self.assertIsInstance(out["details"], dict)


# ---------- split helper ----------

class TestSplitTradesIntoOOSWindows(unittest.TestCase):
    def test_split_returns_disjoint_lists(self):
        trades = []
        for m in range(1, 9):
            for d in range(1, 11):
                trades.append(_trade("2025-{:02d}-{:02d}".format(m, d)))
        out = split_trades_into_oos_windows(trades, validation_months=2)
        # Discovery + validation cover all trades.
        self.assertEqual(
            len(out["discovery_trades"]) + len(out["validation_trades"]),
            len(trades),
        )
        d_dates = {t["date"][:7] for t in out["discovery_trades"]}
        v_dates = {t["date"][:7] for t in out["validation_trades"]}
        self.assertTrue(d_dates.isdisjoint(v_dates))

    def test_validation_strictly_later_than_discovery(self):
        trades = [
            _trade("2025-{:02d}-15".format(m)) for m in range(1, 9)
        ]
        out = split_trades_into_oos_windows(trades, validation_months=2)
        self.assertGreater(
            min(t["date"][:7] for t in out["validation_trades"]),
            max(t["date"][:7] for t in out["discovery_trades"]),
        )

    def test_window_endpoints_returned(self):
        trades = [_trade("2025-{:02d}-15".format(m)) for m in range(1, 9)]
        out = split_trades_into_oos_windows(trades, validation_months=2)
        self.assertIn("discovery_window", out)
        self.assertIn("validation_window", out)
        self.assertEqual(len(out["discovery_window"]), 2)
        self.assertEqual(len(out["validation_window"]), 2)

    def test_insufficient_data_returns_empty(self):
        trades = [_trade("2025-01-01")]
        out = split_trades_into_oos_windows(trades, validation_months=2)
        self.assertEqual(out["validation_trades"], [])

    def test_no_future_data_used_during_discovery(self):
        # The discovery list must not contain any trade whose date is
        # >= the validation_window start.
        trades = [_trade("2025-{:02d}-15".format(m)) for m in range(1, 9)]
        out = split_trades_into_oos_windows(trades, validation_months=2)
        v_start = out["validation_window"][0]
        for t in out["discovery_trades"]:
            self.assertLess(t["date"][:7], v_start)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            split_trades_into_oos_windows("not a list")
        with self.assertRaises(ValueError):
            split_trades_into_oos_windows([], validation_months=0)


# ---------- integration with promote_candidate ----------

class TestIntegrationWithPromoteCandidate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_oos_")
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

    def test_oos_pass_then_promote_succeeds(self):
        d, v = _windows()
        oos = evaluate_oos_promotion(d, v, _report(), _report())
        self.assertTrue(oos["validation_passed"])
        out = promote_candidate(self.store, validation_result=oos)
        self.assertTrue(out["candidate_thresholds_promoted"])

    def test_oos_fail_keeps_active_unchanged(self):
        d, v = _windows()
        oos = evaluate_oos_promotion(
            d, v,
            active_validation_report=_report(profit_factor=2.0),
            candidate_validation_report=_report(profit_factor=0.8),
        )
        self.assertFalse(oos["validation_passed"])
        active_before = self.store.load_active()
        out = promote_candidate(self.store, validation_result=oos)
        self.assertFalse(out["candidate_thresholds_promoted"])
        self.assertEqual(out["reason"], REASON_CANDIDATE_REJECTED_OOS)
        self.assertEqual(self.store.load_active(), active_before)


if __name__ == "__main__":
    unittest.main()
