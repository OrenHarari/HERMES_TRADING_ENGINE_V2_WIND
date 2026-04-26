"""Tests for Prompt 2 / Step 7 - Paper Trading Validation Gate."""

import unittest

from hermes.safety.cost_model import CostModel
from hermes.safety.paper_gate import (
    DETAIL_ACCOUNT_EQUITY_UNAVAILABLE,
    DETAIL_EDGE_DECAY_ACTIVE,
    DETAIL_INSUFFICIENT_PAPER_TRADES,
    DETAIL_INSUFFICIENT_REGIME_DIVERSITY,
    DETAIL_KILL_SWITCH_ACTIVE,
    DETAIL_MAX_DRAWDOWN_TOO_HIGH,
    DETAIL_MISSING_COST_MODEL,
    DETAIL_PROFIT_FACTOR_TOO_LOW,
    PaperGateConfig,
    REASON_LIVE_NOT_ENABLED,
    REASON_PAPER_VALIDATION_FAILED,
    check_live_trade_allowed,
    evaluate_paper_validation,
)


def _good_report(trade_count=150, regimes=("trend_up", "trend_down", "chop")):
    tpr = {r: trade_count // len(regimes) for r in regimes}
    return {
        "trade_count": trade_count,
        "win_rate": 0.55,
        "profit_factor": 1.50,
        "max_drawdown": 200.0,
        "stability_score": 0.50,
        "net_pnl": 1500.0,
        "avg_win": 10.0, "avg_loss": -5.0,
        "trades_per_regime": tpr,
        "cost_model_applied": True,
    }


def _good_inputs(**overrides):
    base = {
        "performance_report": _good_report(),
        "edge_decay_alert": False,
        "kill_switch_active": False,
        "cost_model": CostModel(fee_pct=0.001),
        "account_equity": 100_000.0,
    }
    base.update(overrides)
    return base


# ---------- below 100 paper trades ----------

class TestMinPaperTrades(unittest.TestCase):
    def test_below_100_blocks_live(self):
        out = evaluate_paper_validation(**_good_inputs(
            performance_report=_good_report(trade_count=50),
        ))
        self.assertFalse(out["paper_validation_passed"])
        self.assertFalse(out["live_enabled"])
        self.assertEqual(out["reason"], REASON_PAPER_VALIDATION_FAILED)
        self.assertEqual(out["details"], DETAIL_INSUFFICIENT_PAPER_TRADES)

    def test_at_100_accepted(self):
        out = evaluate_paper_validation(**_good_inputs(
            performance_report=_good_report(trade_count=100),
        ))
        self.assertTrue(out["paper_validation_passed"])

    def test_minimum_is_configurable(self):
        cfg = PaperGateConfig(min_paper_trades=50)
        out = evaluate_paper_validation(
            config=cfg,
            **_good_inputs(performance_report=_good_report(trade_count=60)),
        )
        self.assertTrue(out["paper_validation_passed"])


# ---------- profit_factor floor ----------

class TestProfitFactorFloor(unittest.TestCase):
    def test_below_floor_blocks_live(self):
        cfg = PaperGateConfig(min_profit_factor=1.20)
        out = evaluate_paper_validation(
            config=cfg,
            **_good_inputs(
                performance_report={**_good_report(), "profit_factor": 1.10},
            ),
        )
        self.assertFalse(out["paper_validation_passed"])
        self.assertEqual(out["details"], DETAIL_PROFIT_FACTOR_TOO_LOW)

    def test_at_floor_accepted(self):
        cfg = PaperGateConfig(min_profit_factor=1.20)
        out = evaluate_paper_validation(
            config=cfg,
            **_good_inputs(
                performance_report={**_good_report(), "profit_factor": 1.20},
            ),
        )
        self.assertTrue(out["paper_validation_passed"])

    def test_default_floor_is_1_20_per_spec(self):
        self.assertEqual(PaperGateConfig().min_profit_factor, 1.20)


# ---------- max drawdown ----------

class TestMaxDrawdown(unittest.TestCase):
    def test_above_max_blocks_live(self):
        cfg = PaperGateConfig(max_drawdown_max=100.0)
        out = evaluate_paper_validation(
            config=cfg,
            **_good_inputs(
                performance_report={**_good_report(), "max_drawdown": 150.0},
            ),
        )
        self.assertFalse(out["paper_validation_passed"])
        self.assertEqual(out["details"], DETAIL_MAX_DRAWDOWN_TOO_HIGH)

    def test_at_max_accepted(self):
        cfg = PaperGateConfig(max_drawdown_max=200.0)
        out = evaluate_paper_validation(
            config=cfg,
            **_good_inputs(
                performance_report={**_good_report(), "max_drawdown": 200.0},
            ),
        )
        self.assertTrue(out["paper_validation_passed"])


# ---------- edge decay ----------

class TestEdgeDecay(unittest.TestCase):
    def test_active_edge_decay_blocks_live(self):
        out = evaluate_paper_validation(**_good_inputs(edge_decay_alert=True))
        self.assertFalse(out["paper_validation_passed"])
        self.assertEqual(out["details"], DETAIL_EDGE_DECAY_ACTIVE)


# ---------- kill switch ----------

class TestKillSwitch(unittest.TestCase):
    def test_active_kill_switch_blocks_live(self):
        out = evaluate_paper_validation(**_good_inputs(
            kill_switch_active=True,
        ))
        self.assertFalse(out["paper_validation_passed"])
        self.assertEqual(out["details"], DETAIL_KILL_SWITCH_ACTIVE)


# ---------- cost model ----------

class TestCostModel(unittest.TestCase):
    def test_missing_cost_model_blocks_live(self):
        out = evaluate_paper_validation(**_good_inputs(cost_model=None))
        self.assertFalse(out["paper_validation_passed"])
        self.assertEqual(out["details"], DETAIL_MISSING_COST_MODEL)

    def test_invalid_cost_model_object_blocks_live(self):
        out = evaluate_paper_validation(**_good_inputs(
            cost_model={"fee_pct": 0.001},  # not a CostModel
        ))
        self.assertFalse(out["paper_validation_passed"])
        self.assertEqual(out["details"], DETAIL_MISSING_COST_MODEL)


# ---------- account equity ----------

class TestAccountEquity(unittest.TestCase):
    def test_missing_equity_blocks_live(self):
        out = evaluate_paper_validation(**_good_inputs(account_equity=None))
        self.assertFalse(out["paper_validation_passed"])
        self.assertEqual(out["details"], DETAIL_ACCOUNT_EQUITY_UNAVAILABLE)

    def test_zero_equity_blocks_live(self):
        out = evaluate_paper_validation(**_good_inputs(account_equity=0.0))
        self.assertFalse(out["paper_validation_passed"])
        self.assertEqual(out["details"], DETAIL_ACCOUNT_EQUITY_UNAVAILABLE)


# ---------- regime diversity ----------

class TestRegimeDiversity(unittest.TestCase):
    def test_two_regimes_reports_insufficient_diversity(self):
        report = _good_report(regimes=("trend_up", "chop"))
        out = evaluate_paper_validation(**_good_inputs(
            performance_report=report,
        ))
        # Default config does NOT block on insufficient diversity, but
        # MUST report it.
        self.assertTrue(out["insufficient_regime_diversity"])

    def test_three_regimes_does_not_flag_insufficient_diversity(self):
        out = evaluate_paper_validation(**_good_inputs())
        self.assertFalse(out["insufficient_regime_diversity"])

    def test_config_can_block_on_insufficient_diversity(self):
        cfg = PaperGateConfig(block_on_insufficient_regime_diversity=True)
        report = _good_report(regimes=("trend_up", "chop"))
        out = evaluate_paper_validation(
            config=cfg,
            **_good_inputs(performance_report=report),
        )
        self.assertFalse(out["paper_validation_passed"])
        self.assertEqual(out["details"], DETAIL_INSUFFICIENT_REGIME_DIVERSITY)


# ---------- all-pass case ----------

class TestAllPass(unittest.TestCase):
    def test_all_rules_passing_yields_validation_pass(self):
        out = evaluate_paper_validation(**_good_inputs())
        self.assertTrue(out["paper_validation_passed"])
        self.assertEqual(out["reason"], "")
        self.assertEqual(out["details"], "")

    def test_paper_pass_does_not_auto_enable_live(self):
        # Spec: passing paper validation does not auto-enable real trading.
        out = evaluate_paper_validation(**_good_inputs())
        # The evaluator never sets live_enabled=True by itself.
        self.assertFalse(out["live_enabled"])

    def test_output_canonical_keys(self):
        out = evaluate_paper_validation(**_good_inputs())
        self.assertEqual(
            set(out.keys()),
            {
                "paper_validation_passed",
                "live_enabled",
                "reason",
                "details",
                "insufficient_regime_diversity",
            },
        )


# ---------- check_live_trade_allowed ----------

class TestLiveTradeAllowed(unittest.TestCase):
    def _passing_paper_result(self):
        return evaluate_paper_validation(**_good_inputs())

    def test_blocks_when_live_enabled_missing(self):
        out = check_live_trade_allowed(
            paper_validation_result=self._passing_paper_result(),
            live_enabled=False,
            kill_switch_active=False,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_LIVE_NOT_ENABLED)

    def test_blocks_when_live_enabled_is_none(self):
        out = check_live_trade_allowed(
            paper_validation_result=self._passing_paper_result(),
            live_enabled=None,
            kill_switch_active=False,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_LIVE_NOT_ENABLED)

    def test_blocks_when_paper_validation_failed(self):
        failed = evaluate_paper_validation(**_good_inputs(
            performance_report=_good_report(trade_count=10),
        ))
        out = check_live_trade_allowed(
            paper_validation_result=failed,
            live_enabled=True,
            kill_switch_active=False,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_PAPER_VALIDATION_FAILED)

    def test_blocks_when_kill_switch_active(self):
        out = check_live_trade_allowed(
            paper_validation_result=self._passing_paper_result(),
            live_enabled=True,
            kill_switch_active=True,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], "kill_switch_active")

    def test_allows_when_all_rules_pass(self):
        out = check_live_trade_allowed(
            paper_validation_result=self._passing_paper_result(),
            live_enabled=True,
            kill_switch_active=False,
        )
        self.assertTrue(out["trade_allowed"])
        self.assertEqual(out["reason"], "")

    def test_canonical_output_keys(self):
        out = check_live_trade_allowed(
            paper_validation_result=self._passing_paper_result(),
            live_enabled=True,
            kill_switch_active=False,
        )
        self.assertEqual(set(out.keys()), {"trade_allowed", "reason"})


# ---------- determinism + input validation ----------

class TestDeterminismAndInputs(unittest.TestCase):
    def test_deterministic(self):
        a = evaluate_paper_validation(**_good_inputs())
        b = evaluate_paper_validation(**_good_inputs())
        self.assertEqual(a, b)

    def test_rejects_invalid_performance_report(self):
        with self.assertRaises(ValueError):
            evaluate_paper_validation(
                performance_report="not a dict",
                edge_decay_alert=False,
                kill_switch_active=False,
                cost_model=CostModel(),
                account_equity=100.0,
            )

    def test_rejects_invalid_kill_switch_flag(self):
        with self.assertRaises(ValueError):
            evaluate_paper_validation(**_good_inputs(kill_switch_active="yes"))


if __name__ == "__main__":
    unittest.main()
