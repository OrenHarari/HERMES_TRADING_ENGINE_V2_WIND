"""Tests for Prompt 2 / Step 5C - Cost, Slippage, Execution Model."""

import unittest

from hermes.decision.performance import compute_performance_report
from hermes.safety.cost_model import (
    REASON_MISSING_COST_MODEL,
    CostModel,
    apply_cost_model_to_trade,
    check_cost_model_for_live,
    validate_cost_model,
)


class TestCostModelConstruction(unittest.TestCase):
    def test_defaults_are_zero(self):
        cm = CostModel()
        self.assertEqual(cm.fee_pct, 0.0)
        self.assertEqual(cm.slippage_pct, 0.0)
        self.assertEqual(cm.spread_pct, 0.0)

    def test_rejects_negative_fee_pct(self):
        with self.assertRaises(ValueError):
            CostModel(fee_pct=-0.001)

    def test_rejects_negative_slippage_pct(self):
        with self.assertRaises(ValueError):
            CostModel(slippage_pct=-0.001)

    def test_rejects_negative_spread_pct(self):
        with self.assertRaises(ValueError):
            CostModel(spread_pct=-0.001)

    def test_rejects_unreasonable_fee_pct(self):
        # > 100% per-side notional is pathological -> reject.
        with self.assertRaises(ValueError):
            CostModel(fee_pct=2.0)

    def test_rejects_unknown_entry_fill_model(self):
        with self.assertRaises(ValueError):
            CostModel(entry_fill_model="quantum_teleport")

    def test_rejects_unknown_exit_fill_model(self):
        with self.assertRaises(ValueError):
            CostModel(exit_fill_model="quantum_teleport")

    def test_rejects_non_numeric(self):
        with self.assertRaises(ValueError):
            CostModel(fee_pct="0.001")
        with self.assertRaises(ValueError):
            CostModel(slippage_pct=True)


class TestCostModelApply(unittest.TestCase):
    def test_zero_costs_make_net_equal_gross(self):
        cm = CostModel()
        out = cm.apply(entry_price=100.0, exit_price=110.0, position_size=10.0)
        self.assertEqual(out["gross_pnl"], 100.0)
        self.assertEqual(out["net_pnl"], 100.0)
        self.assertEqual(out["fees"], 0.0)
        self.assertEqual(out["slippage"], 0.0)
        self.assertEqual(out["spread_cost"], 0.0)

    def test_fees_reduce_net_pnl(self):
        cm = CostModel(fee_pct=0.001)  # 10 bps each-side notional
        out = cm.apply(100.0, 110.0, 10.0)
        # notional = (100 + 110) * 10 = 2100; fees = 2.1
        self.assertAlmostEqual(out["fees"], 2.1, places=10)
        self.assertAlmostEqual(out["net_pnl"], out["gross_pnl"] - 2.1, places=10)

    def test_slippage_reduces_net_pnl(self):
        cm = CostModel(slippage_pct=0.0005)
        out = cm.apply(100.0, 110.0, 10.0)
        self.assertAlmostEqual(out["slippage"], 0.0005 * 2100.0, places=10)
        self.assertAlmostEqual(
            out["net_pnl"], out["gross_pnl"] - out["slippage"], places=10
        )

    def test_spread_reduces_net_pnl(self):
        cm = CostModel(spread_pct=0.0002)
        out = cm.apply(100.0, 110.0, 10.0)
        self.assertAlmostEqual(out["spread_cost"], 0.0002 * 2100.0, places=10)

    def test_all_components_combine(self):
        cm = CostModel(fee_pct=0.001, slippage_pct=0.0005, spread_pct=0.0002)
        out = cm.apply(100.0, 110.0, 10.0)
        expected_costs = (0.001 + 0.0005 + 0.0002) * 2100.0
        self.assertAlmostEqual(
            out["net_pnl"], out["gross_pnl"] - expected_costs, places=10
        )

    def test_gross_pnl_differs_from_net_when_costs_exist(self):
        # Spec: "gross_pnl != net_pnl when fees/slippage exist"
        cm = CostModel(fee_pct=0.001)
        out = cm.apply(100.0, 110.0, 10.0)
        self.assertNotEqual(out["gross_pnl"], out["net_pnl"])

    def test_losing_trade_costs_make_net_more_negative(self):
        cm = CostModel(fee_pct=0.001)
        out = cm.apply(entry_price=100.0, exit_price=95.0, position_size=10.0)
        self.assertLess(out["net_pnl"], out["gross_pnl"])

    def test_apply_rejects_non_positive_prices(self):
        cm = CostModel()
        with self.assertRaises(ValueError):
            cm.apply(0.0, 100.0, 10.0)
        with self.assertRaises(ValueError):
            cm.apply(100.0, -1.0, 10.0)

    def test_apply_rejects_negative_position_size(self):
        cm = CostModel()
        with self.assertRaises(ValueError):
            cm.apply(100.0, 110.0, -5.0)

    def test_apply_zero_position_size_allowed(self):
        cm = CostModel(fee_pct=0.001)
        out = cm.apply(100.0, 110.0, 0.0)
        self.assertEqual(out["gross_pnl"], 0.0)
        self.assertEqual(out["net_pnl"], 0.0)
        self.assertEqual(out["fees"], 0.0)

    def test_output_includes_fill_model_labels(self):
        cm = CostModel(entry_fill_model="quoted", exit_fill_model="with_slippage")
        out = cm.apply(100.0, 110.0, 1.0)
        self.assertEqual(out["entry_fill_model"], "quoted")
        self.assertEqual(out["exit_fill_model"], "with_slippage")

    def test_output_keys_canonical(self):
        cm = CostModel()
        out = cm.apply(100.0, 110.0, 1.0)
        self.assertEqual(
            set(out.keys()),
            {
                "gross_pnl", "fees", "slippage", "spread_cost", "net_pnl",
                "entry_fill_model", "exit_fill_model",
            },
        )


class TestCostModelDeterminism(unittest.TestCase):
    def test_same_inputs_same_output(self):
        cm = CostModel(fee_pct=0.001, slippage_pct=0.0005, spread_pct=0.0002)
        a = cm.apply(123.45, 678.90, 7.0)
        b = cm.apply(123.45, 678.90, 7.0)
        self.assertEqual(a, b)

    def test_repeated_apply_produces_identical_breakdowns(self):
        cm = CostModel(fee_pct=0.001)
        results = [cm.apply(100.0, 110.0, 1.0) for _ in range(5)]
        self.assertEqual(len(set(tuple(sorted(r.items())) for r in results)), 1)


class TestValidateCostModel(unittest.TestCase):
    def test_missing_cost_model_blocks_validation(self):
        out = validate_cost_model(None)
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], REASON_MISSING_COST_MODEL)

    def test_non_costmodel_object_blocks_validation(self):
        out = validate_cost_model("not_a_cost_model")
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], REASON_MISSING_COST_MODEL)

    def test_valid_cost_model_passes(self):
        cm = CostModel(fee_pct=0.001)
        out = validate_cost_model(cm)
        self.assertTrue(out["validation_passed"])
        self.assertEqual(out["reason"], "")

    def test_validation_output_shape(self):
        out = validate_cost_model(CostModel())
        self.assertEqual(set(out.keys()), {"validation_passed", "reason"})


class TestCostModelLiveGate(unittest.TestCase):
    def test_missing_cost_model_blocks_live_trading(self):
        out = check_cost_model_for_live(None)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_MISSING_COST_MODEL)

    def test_non_costmodel_object_blocks_live_trading(self):
        out = check_cost_model_for_live({"fee_pct": 0.001})
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_MISSING_COST_MODEL)

    def test_valid_cost_model_allows_live_trading(self):
        out = check_cost_model_for_live(CostModel(fee_pct=0.001))
        self.assertTrue(out["trade_allowed"])
        self.assertEqual(out["reason"], "")

    def test_live_gate_output_shape(self):
        out = check_cost_model_for_live(None)
        self.assertEqual(set(out.keys()), {"trade_allowed", "reason"})


class TestApplyCostModelToTrade(unittest.TestCase):
    """Helper that augments a completed-trade record with cost fields."""

    def test_fills_gross_pnl_fees_slippage_spread_net_pnl(self):
        cm = CostModel(fee_pct=0.001)
        trade = {
            "entry_price": 100.0, "exit_price": 110.0, "position_size": 10.0,
            "pnl": 100.0,
        }
        out = apply_cost_model_to_trade(trade, cm)
        self.assertIn("net_pnl", out)
        self.assertIn("fees", out)
        self.assertIn("slippage", out)
        self.assertIn("spread_cost", out)
        self.assertNotEqual(out["net_pnl"], out["gross_pnl"])

    def test_does_not_mutate_input(self):
        cm = CostModel(fee_pct=0.001)
        trade = {
            "entry_price": 100.0, "exit_price": 110.0, "position_size": 1.0,
            "pnl": 10.0,
        }
        trade_before = dict(trade)
        apply_cost_model_to_trade(trade, cm)
        self.assertEqual(trade, trade_before)

    def test_rejects_missing_required_fields(self):
        cm = CostModel()
        with self.assertRaises(ValueError):
            apply_cost_model_to_trade({"entry_price": 1.0}, cm)


class TestPerformanceReportCostModelAppliedFlag(unittest.TestCase):
    """Spec: 'performance report marks cost_model_applied correctly'."""

    def test_default_remains_false_for_backward_compatibility(self):
        # Existing Prompt 1 callers must still see cost_model_applied=False.
        rep = compute_performance_report([{"net_pnl": 1.0}])
        self.assertFalse(rep["cost_model_applied"])

    def test_explicit_true_marks_report(self):
        rep = compute_performance_report(
            [{"net_pnl": 1.0}], cost_model_applied=True
        )
        self.assertTrue(rep["cost_model_applied"])

    def test_explicit_false_keeps_false(self):
        rep = compute_performance_report(
            [{"net_pnl": 1.0}], cost_model_applied=False
        )
        self.assertFalse(rep["cost_model_applied"])

    def test_empty_trades_with_flag_true_still_reports_true(self):
        rep = compute_performance_report([], cost_model_applied=True)
        self.assertTrue(rep["cost_model_applied"])

    def test_net_pnl_never_ignores_fees(self):
        # When cost_model is applied to trades, net_pnl in the records must
        # already reflect costs. Performance report must use those values.
        cm = CostModel(fee_pct=0.001)
        raw = [
            {"entry_price": 100.0, "exit_price": 110.0,
             "position_size": 10.0, "pnl": 100.0},
            {"entry_price": 100.0, "exit_price": 95.0,
             "position_size": 10.0, "pnl": -50.0},
        ]
        adjusted = [apply_cost_model_to_trade(t, cm) for t in raw]
        rep = compute_performance_report(adjusted, cost_model_applied=True)
        # Sum of net_pnl < sum of gross_pnl because of fees on every trade.
        gross = sum(t["gross_pnl"] for t in adjusted)
        self.assertLess(rep["net_pnl"], gross)


if __name__ == "__main__":
    unittest.main()
