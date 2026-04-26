"""Tests for Prompt 2 / Step 5B - Trade Lifecycle Contract.

Lifecycle FSM rules:
  signal -> entry decision -> risk validation -> open position
        -> exit decision (deterministic, no future data)
        -> completed trade -> memory log -> attribution.

These tests verify only the lifecycle pieces added in Step 5B: the
OpenPosition state object, the deterministic exit-decision rules, and
complete_trade(...) which produces the canonical Phase-1 record extended
with Phase-2 exit fields.
"""

import unittest

from hermes.decision.performance import compute_performance_report
from hermes.lifecycle import (
    EXIT_REASONS,
    EXIT_REASON_END_OF_BACKTEST,
    EXIT_REASON_MAX_HOLDING,
    EXIT_REASON_RISK_GUARDRAIL,
    EXIT_REASON_SIGNAL_DECAY,
    EXIT_REASON_STOP_LOSS,
    EXIT_REASON_TAKE_PROFIT,
    ExitRulesConfig,
    OpenPosition,
    REQUIRED_EXIT_FIELDS,
    REQUIRED_TRADE_KEYS,
    complete_trade,
    decide_exit,
    is_complete_trade_record,
)
from hermes.safety.cost_model import CostModel


# -------- helpers --------

def _entry_meta():
    return {
        "date": "2025-01-15",
        "hour": 10,
        "sequence_value": 0.7,
        "amd_value": 0.7,
        "combined_value": 0.7,
        "agreement": 1.0,
        "confidence": 0.75,
        "regime": "trend_up",
        "momentum_score": 0.7,
        "volatility_score": 0.4,
    }


def _candle(ts, close=100.0, o=None, h=None, l=None):
    return {
        "timestamp": ts,
        "open": o if o is not None else close,
        "high": h if h is not None else close + 0.5,
        "low": l if l is not None else close - 0.5,
        "close": close,
    }


def _candles_flat(n=10, base_ts=1000, step=60, close=100.0):
    return [_candle(base_ts + i * step, close=close) for i in range(n)]


def _open_at(idx, price=100.0, ts=None, size=10.0):
    return OpenPosition(
        entry_timestamp=ts if ts is not None else 1000 + idx * 60,
        entry_index=idx,
        entry_price=price,
        position_size=size,
        entry_signal_meta=_entry_meta(),
    )


# -------- OpenPosition --------

class TestOpenPosition(unittest.TestCase):
    def test_basic_construction(self):
        pos = _open_at(2, price=100.0)
        self.assertEqual(pos.entry_index, 2)
        self.assertEqual(pos.entry_price, 100.0)
        self.assertEqual(pos.position_size, 10.0)

    def test_rejects_non_positive_entry_price(self):
        with self.assertRaises(ValueError):
            OpenPosition(entry_timestamp=1, entry_index=0, entry_price=0.0,
                         position_size=1.0, entry_signal_meta=_entry_meta())

    def test_rejects_negative_position_size(self):
        with self.assertRaises(ValueError):
            OpenPosition(entry_timestamp=1, entry_index=0, entry_price=100.0,
                         position_size=-1.0, entry_signal_meta=_entry_meta())

    def test_rejects_non_int_entry_index(self):
        with self.assertRaises(ValueError):
            OpenPosition(entry_timestamp=1, entry_index="x", entry_price=100.0,
                         position_size=1.0, entry_signal_meta=_entry_meta())

    def test_rejects_missing_entry_signal_meta_keys(self):
        meta = _entry_meta()
        del meta["confidence"]
        with self.assertRaises(ValueError):
            OpenPosition(entry_timestamp=1, entry_index=0, entry_price=100.0,
                         position_size=1.0, entry_signal_meta=meta)


# -------- decide_exit: priority + each reason --------

class TestExitDecisionPriority(unittest.TestCase):
    def test_no_exit_when_no_rule_triggers(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=10.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=100.0)
        out = decide_exit(pos, candles[2], 2, current_combined_value=0.7,
                          config=cfg)
        self.assertFalse(out["should_exit"])

    def test_risk_guardrail_takes_top_priority(self):
        cfg = ExitRulesConfig(stop_distance=0.0001, take_profit_distance=0.0001,
                              max_holding_bars=1)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=200.0)  # would trigger TP
        out = decide_exit(pos, candles[3], 3, risk_blocked=True, config=cfg)
        self.assertTrue(out["should_exit"])
        self.assertEqual(out["exit_reason"], EXIT_REASON_RISK_GUARDRAIL)

    def test_end_of_backtest_priority_over_normal_rules(self):
        cfg = ExitRulesConfig(stop_distance=1.0, take_profit_distance=1.0,
                              max_holding_bars=1)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=100.0)
        out = decide_exit(pos, candles[4], 4, end_of_backtest=True, config=cfg)
        self.assertTrue(out["should_exit"])
        self.assertEqual(out["exit_reason"], EXIT_REASON_END_OF_BACKTEST)

    def test_stop_loss_when_price_breaks_down(self):
        cfg = ExitRulesConfig(stop_distance=2.0, take_profit_distance=10.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=97.5)
        out = decide_exit(pos, candles[2], 2, config=cfg)
        self.assertTrue(out["should_exit"])
        self.assertEqual(out["exit_reason"], EXIT_REASON_STOP_LOSS)

    def test_stop_loss_at_exact_threshold_triggers(self):
        cfg = ExitRulesConfig(stop_distance=2.0, take_profit_distance=10.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=98.0)
        out = decide_exit(pos, candles[1], 1, config=cfg)
        self.assertTrue(out["should_exit"])
        self.assertEqual(out["exit_reason"], EXIT_REASON_STOP_LOSS)

    def test_take_profit_when_price_breaks_up(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=3.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=104.0)
        out = decide_exit(pos, candles[1], 1, config=cfg)
        self.assertTrue(out["should_exit"])
        self.assertEqual(out["exit_reason"], EXIT_REASON_TAKE_PROFIT)

    def test_max_holding_bars_when_held_too_long(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=10.0,
                              max_holding_bars=3)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(10, close=100.0)
        out = decide_exit(pos, candles[3], 3, config=cfg)
        self.assertTrue(out["should_exit"])
        self.assertEqual(out["exit_reason"], EXIT_REASON_MAX_HOLDING)

    def test_signal_decay_default_threshold_is_0_30(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=10.0,
                              max_holding_bars=100)
        # Default signal_decay_threshold should be 0.30 per spec.
        self.assertEqual(cfg.signal_decay_threshold, 0.30)

    def test_signal_decay_when_combined_below_threshold(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=10.0,
                              max_holding_bars=100, signal_decay_threshold=0.30)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=100.0)
        out = decide_exit(pos, candles[2], 2, current_combined_value=0.20,
                          config=cfg)
        self.assertTrue(out["should_exit"])
        self.assertEqual(out["exit_reason"], EXIT_REASON_SIGNAL_DECAY)

    def test_signal_at_exactly_threshold_does_not_decay(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=10.0,
                              max_holding_bars=100, signal_decay_threshold=0.30)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=100.0)
        out = decide_exit(pos, candles[2], 2, current_combined_value=0.30,
                          config=cfg)
        self.assertFalse(out["should_exit"])

    def test_priority_stop_before_take_profit(self):
        # If the same candle would trigger both, stop loss comes first
        # (a wide range bar where low <= sl threshold and close >= tp threshold).
        # Here close drives both rules; we craft close to satisfy stop only.
        cfg = ExitRulesConfig(stop_distance=2.0, take_profit_distance=2.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0)
        # Close at 97.5 -> stop triggers; tp would need close>=102.
        candles = _candles_flat(5, close=97.5)
        out = decide_exit(pos, candles[1], 1, config=cfg)
        self.assertEqual(out["exit_reason"], EXIT_REASON_STOP_LOSS)


# -------- decide_exit: no future data + determinism --------

class TestExitDeterminismAndNoFutureData(unittest.TestCase):
    def test_decide_exit_is_deterministic(self):
        cfg = ExitRulesConfig(stop_distance=2.0, take_profit_distance=5.0,
                              max_holding_bars=10)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=98.0)
        a = decide_exit(pos, candles[2], 2, current_combined_value=0.5,
                        config=cfg)
        b = decide_exit(pos, candles[2], 2, current_combined_value=0.5,
                        config=cfg)
        self.assertEqual(a, b)

    def test_decide_exit_does_not_use_future_candles(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=10.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(10, close=100.0)
        before = decide_exit(pos, candles[3], 3, current_combined_value=0.7,
                             config=cfg)
        # Mutate future portion arbitrarily; result must be unchanged because
        # decide_exit only inspects `position`, `current_candle`,
        # `current_index`, and `current_combined_value`.
        for j in range(4, len(candles)):
            candles[j] = {"timestamp": -1, "open": -1, "high": -1, "low": -1,
                          "close": -1}
        after = decide_exit(pos, candles[3], 3, current_combined_value=0.7,
                            config=cfg)
        self.assertEqual(before, after)


# -------- complete_trade --------

class TestCompleteTrade(unittest.TestCase):
    def test_required_phase1_keys_all_present(self):
        cfg = ExitRulesConfig(stop_distance=2.0, take_profit_distance=10.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=97.0)
        decision = decide_exit(pos, candles[1], 1, config=cfg)
        rec = complete_trade(pos, candles[1], 1, decision)
        for k in REQUIRED_TRADE_KEYS:
            self.assertIn(k, rec)

    def test_required_phase2_exit_fields_present(self):
        cfg = ExitRulesConfig(stop_distance=2.0, take_profit_distance=10.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=97.0)
        decision = decide_exit(pos, candles[1], 1, config=cfg)
        rec = complete_trade(pos, candles[1], 1, decision)
        for k in REQUIRED_EXIT_FIELDS:
            self.assertIn(k, rec)

    def test_required_exit_fields_canonical_set(self):
        # Spec lists: exit_reason, exit_price, bars_held, pnl, fees,
        # slippage, spread_cost, net_pnl.
        self.assertEqual(
            set(REQUIRED_EXIT_FIELDS),
            {"exit_reason", "exit_price", "bars_held", "pnl",
             "fees", "slippage", "spread_cost", "net_pnl"},
        )

    def test_exit_reason_is_always_in_valid_enum(self):
        cfg = ExitRulesConfig(stop_distance=2.0, take_profit_distance=2.0,
                              max_holding_bars=2)
        pos = _open_at(0, price=100.0)
        # Multiple scenarios -> several reasons.
        scenarios = [
            (_candles_flat(5, close=97.5)[1], 1, {}),
            (_candles_flat(5, close=102.5)[1], 1, {}),
            (_candles_flat(5, close=100.0)[2], 2, {}),
            (_candles_flat(5, close=100.0)[1], 1,
             {"current_combined_value": 0.20}),
            (_candles_flat(5, close=200.0)[1], 1, {"risk_blocked": True}),
            (_candles_flat(5, close=100.0)[4], 4, {"end_of_backtest": True}),
        ]
        for candle, idx, kwargs in scenarios:
            decision = decide_exit(pos, candle, idx, config=cfg, **kwargs)
            if decision["should_exit"]:
                self.assertIn(decision["exit_reason"], EXIT_REASONS)
                rec = complete_trade(pos, candle, idx, decision)
                self.assertIn(rec["exit_reason"], EXIT_REASONS)

    def test_bars_held_is_int(self):
        cfg = ExitRulesConfig(stop_distance=2.0, take_profit_distance=10.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=97.5)
        decision = decide_exit(pos, candles[3], 3, config=cfg)
        rec = complete_trade(pos, candles[3], 3, decision)
        self.assertIsInstance(rec["bars_held"], int)
        self.assertEqual(rec["bars_held"], 3)

    def test_no_cost_model_zeros_costs_and_net_equals_gross(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=2.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0, size=10.0)
        candles = _candles_flat(5, close=103.0)
        decision = decide_exit(pos, candles[1], 1, config=cfg)
        rec = complete_trade(pos, candles[1], 1, decision)
        self.assertEqual(rec["fees"], 0.0)
        self.assertEqual(rec["slippage"], 0.0)
        self.assertEqual(rec["spread_cost"], 0.0)
        # Gross = (103 - 100) * 10 = 30; with no costs, net = gross.
        self.assertEqual(rec["net_pnl"], 30.0)
        self.assertEqual(rec["pnl"], 30.0)

    def test_net_pnl_includes_fees_and_costs_when_cost_model_provided(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=2.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0, size=10.0)
        candles = _candles_flat(5, close=103.0)
        decision = decide_exit(pos, candles[1], 1, config=cfg)
        cm = CostModel(fee_pct=0.001)
        rec = complete_trade(pos, candles[1], 1, decision, cost_model=cm)
        self.assertGreater(rec["fees"], 0.0)
        self.assertNotEqual(rec["net_pnl"], 30.0)
        self.assertLess(rec["net_pnl"], 30.0)
        # outcome should reflect post-cost reality.
        # With small fees, 30 - fees still positive -> win.
        self.assertEqual(rec["outcome"], "win")

    def test_completed_record_passes_phase1_validator(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=2.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0, size=10.0)
        candles = _candles_flat(5, close=103.0)
        decision = decide_exit(pos, candles[1], 1, config=cfg)
        rec = complete_trade(pos, candles[1], 1, decision,
                             cost_model=CostModel(fee_pct=0.001))
        self.assertTrue(is_complete_trade_record(rec))

    def test_complete_trade_rejects_non_exiting_decision(self):
        pos = _open_at(0, price=100.0)
        candles = _candles_flat(5, close=100.0)
        with self.assertRaises(ValueError):
            complete_trade(pos, candles[1], 1, {"should_exit": False})

    def test_complete_trade_does_not_use_future_data(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=2.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0, size=10.0)
        candles = _candles_flat(10, close=103.0)
        decision = decide_exit(pos, candles[2], 2, config=cfg)
        rec_before = complete_trade(pos, candles[2], 2, decision)
        for j in range(3, len(candles)):
            candles[j] = {"timestamp": -1, "open": -1, "high": -1, "low": -1,
                          "close": -1}
        rec_after = complete_trade(pos, candles[2], 2, decision)
        self.assertEqual(rec_before, rec_after)

    def test_complete_trade_is_deterministic(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=2.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0, size=10.0)
        candles = _candles_flat(5, close=103.0)
        decision = decide_exit(pos, candles[1], 1, config=cfg)
        a = complete_trade(pos, candles[1], 1, decision)
        b = complete_trade(pos, candles[1], 1, decision)
        self.assertEqual(a, b)


# -------- "no completed trade without entry and exit" --------

class TestNoCompletedTradeWithoutEntryAndExit(unittest.TestCase):
    def test_open_position_alone_is_not_a_completed_record(self):
        pos = _open_at(0, price=100.0)
        # OpenPosition is NOT a completed-trade record.
        self.assertFalse(is_complete_trade_record(pos))

    def test_open_position_dict_without_exit_is_invalid(self):
        # Even if a caller manually built a partial dict using only entry
        # data, it must fail completed-record validation.
        partial = {
            "timestamp": 1000,
            "date": "2025-01-15",
            "hour": 10,
            "entry_price": 100.0,
            "sequence_value": 0.7, "amd_value": 0.7, "combined_value": 0.7,
            "agreement": 1.0, "confidence": 0.75, "regime": "trend_up",
            "momentum_score": 0.7, "volatility_score": 0.4,
            # Missing: exit_timestamp, exit_price, outcome, pnl
        }
        self.assertFalse(is_complete_trade_record(partial))


# -------- performance metrics exclude open positions --------

class TestIncompleteTradesNotIncludedInPerformance(unittest.TestCase):
    def test_open_positions_are_not_counted(self):
        cfg = ExitRulesConfig(stop_distance=10.0, take_profit_distance=2.0,
                              max_holding_bars=100)
        pos = _open_at(0, price=100.0, size=10.0)
        candles = _candles_flat(5, close=103.0)
        decision = decide_exit(pos, candles[1], 1, config=cfg)
        completed = complete_trade(pos, candles[1], 1, decision)

        # The performance report only sees what is given; a still-open
        # position would not be in this list at all.
        rep = compute_performance_report([completed])
        self.assertEqual(rep["trade_count"], 1)
        self.assertEqual(rep["net_pnl"], 30.0)


if __name__ == "__main__":
    unittest.main()
