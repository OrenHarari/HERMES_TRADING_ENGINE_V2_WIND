"""Tests for Prompt 1 / Step 5 Part 3 - Risk Guardrails."""

import unittest

from hermes.risk import (
    REASON_COOLDOWN,
    REASON_DAILY_LOSS,
    REASON_MAX_CONSECUTIVE_LOSSES,
    REASON_MAX_TRADES,
    RiskConfig,
    RiskState,
)


class TestRiskConfig(unittest.TestCase):
    def test_defaults(self):
        c = RiskConfig()
        self.assertEqual(c.max_trades_per_day, 10)
        self.assertEqual(c.max_daily_loss, 1000.0)
        self.assertEqual(c.max_consecutive_losses, 4)
        self.assertEqual(c.cooldown_seconds, 60)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            RiskConfig(max_trades_per_day=-1)
        with self.assertRaises(ValueError):
            RiskConfig(max_daily_loss=-100.0)
        with self.assertRaises(ValueError):
            RiskConfig(max_consecutive_losses=-1)
        with self.assertRaises(ValueError):
            RiskConfig(cooldown_seconds=-1)


class TestRiskStateBasic(unittest.TestCase):
    def test_fresh_state_allows_first_trade(self):
        rs = RiskState()
        out = rs.check(now_ts=0, day_key="d1")
        self.assertTrue(out["allowed"])
        self.assertEqual(out["reason"], "")

    def test_check_is_idempotent(self):
        rs = RiskState()
        a = rs.check(now_ts=0, day_key="d1")
        b = rs.check(now_ts=0, day_key="d1")
        self.assertEqual(a, b)
        self.assertEqual(rs.trades_today, 0)


class TestMaxTradesPerDay(unittest.TestCase):
    def test_blocks_after_max(self):
        rs = RiskState(RiskConfig(max_trades_per_day=2, cooldown_seconds=0))
        rs.register_trade_entry(0, "d1")
        rs.register_trade_entry(0, "d1")
        out = rs.check(now_ts=0, day_key="d1")
        self.assertFalse(out["allowed"])
        self.assertEqual(out["reason"], REASON_MAX_TRADES)

    def test_resets_on_new_day(self):
        rs = RiskState(RiskConfig(max_trades_per_day=2, cooldown_seconds=0))
        rs.register_trade_entry(0, "d1")
        rs.register_trade_entry(0, "d1")
        # New day - check should allow.
        out = rs.check(now_ts=0, day_key="d2")
        self.assertTrue(out["allowed"])

    def test_zero_max_trades_blocks_immediately(self):
        rs = RiskState(RiskConfig(max_trades_per_day=0))
        out = rs.check(now_ts=0, day_key="d1")
        self.assertFalse(out["allowed"])
        self.assertEqual(out["reason"], REASON_MAX_TRADES)


class TestMaxDailyLoss(unittest.TestCase):
    def test_blocks_when_daily_loss_reached(self):
        rs = RiskState(RiskConfig(max_daily_loss=100.0, cooldown_seconds=0))
        rs.register_trade_outcome(-50.0, "d1")
        rs.register_trade_outcome(-60.0, "d1")  # daily_pnl = -110
        out = rs.check(now_ts=0, day_key="d1")
        self.assertFalse(out["allowed"])
        self.assertEqual(out["reason"], REASON_DAILY_LOSS)

    def test_does_not_block_below_loss_limit(self):
        rs = RiskState(RiskConfig(max_daily_loss=100.0, cooldown_seconds=0))
        rs.register_trade_outcome(-50.0, "d1")
        out = rs.check(now_ts=0, day_key="d1")
        self.assertTrue(out["allowed"])

    def test_wins_alone_do_not_increase_risk_allowance(self):
        rs = RiskState(RiskConfig(max_trades_per_day=2, cooldown_seconds=0))
        rs.register_trade_entry(0, "d1")
        rs.register_trade_outcome(500.0, "d1")  # big win
        rs.register_trade_entry(0, "d1")
        rs.register_trade_outcome(500.0, "d1")  # another big win
        # Already 2 trades this day -> blocked, despite wins.
        out = rs.check(now_ts=0, day_key="d1")
        self.assertFalse(out["allowed"])
        self.assertEqual(out["reason"], REASON_MAX_TRADES)


class TestConsecutiveLosses(unittest.TestCase):
    def test_blocks_after_max_consecutive_losses(self):
        rs = RiskState(RiskConfig(max_consecutive_losses=3, cooldown_seconds=0))
        rs.register_trade_outcome(-10.0, "d1")
        rs.register_trade_outcome(-10.0, "d1")
        rs.register_trade_outcome(-10.0, "d1")
        out = rs.check(now_ts=0, day_key="d1")
        self.assertFalse(out["allowed"])
        self.assertEqual(out["reason"], REASON_MAX_CONSECUTIVE_LOSSES)

    def test_win_resets_counter(self):
        rs = RiskState(RiskConfig(max_consecutive_losses=3, cooldown_seconds=0))
        rs.register_trade_outcome(-10.0, "d1")
        rs.register_trade_outcome(-10.0, "d1")
        rs.register_trade_outcome(+5.0, "d1")  # win resets
        rs.register_trade_outcome(-10.0, "d1")
        rs.register_trade_outcome(-10.0, "d1")
        out = rs.check(now_ts=0, day_key="d1")
        self.assertTrue(out["allowed"])

    def test_breakeven_resets_counter(self):
        rs = RiskState(RiskConfig(max_consecutive_losses=2, cooldown_seconds=0))
        rs.register_trade_outcome(-10.0, "d1")
        rs.register_trade_outcome(0.0, "d1")  # breakeven resets
        out = rs.check(now_ts=0, day_key="d1")
        self.assertTrue(out["allowed"])
        self.assertEqual(rs.consecutive_losses, 0)

    def test_streak_persists_across_days(self):
        rs = RiskState(RiskConfig(max_consecutive_losses=2, cooldown_seconds=0))
        rs.register_trade_outcome(-10.0, "d1")
        rs.register_trade_outcome(-10.0, "d2")
        out = rs.check(now_ts=0, day_key="d2")
        self.assertFalse(out["allowed"])
        self.assertEqual(out["reason"], REASON_MAX_CONSECUTIVE_LOSSES)


class TestCooldown(unittest.TestCase):
    def test_blocks_inside_cooldown_window(self):
        rs = RiskState(RiskConfig(cooldown_seconds=60))
        rs.register_trade_entry(now_ts=100, day_key="d1")
        out = rs.check(now_ts=130, day_key="d1")
        self.assertFalse(out["allowed"])
        self.assertEqual(out["reason"], REASON_COOLDOWN)

    def test_allows_after_cooldown(self):
        rs = RiskState(RiskConfig(cooldown_seconds=60))
        rs.register_trade_entry(now_ts=100, day_key="d1")
        out = rs.check(now_ts=200, day_key="d1")
        self.assertTrue(out["allowed"])


class TestPriorityOrder(unittest.TestCase):
    def test_max_trades_takes_precedence_over_cooldown(self):
        rs = RiskState(RiskConfig(max_trades_per_day=1, cooldown_seconds=60))
        rs.register_trade_entry(now_ts=0, day_key="d1")
        out = rs.check(now_ts=10, day_key="d1")
        self.assertFalse(out["allowed"])
        self.assertEqual(out["reason"], REASON_MAX_TRADES)

    def test_daily_loss_precedence_over_consecutive_losses(self):
        rs = RiskState(
            RiskConfig(
                max_trades_per_day=100,
                max_daily_loss=50.0,
                max_consecutive_losses=10,
                cooldown_seconds=0,
            )
        )
        rs.register_trade_outcome(-60.0, "d1")  # daily loss reached
        out = rs.check(now_ts=0, day_key="d1")
        self.assertFalse(out["allowed"])
        self.assertEqual(out["reason"], REASON_DAILY_LOSS)


class TestRegisterValidation(unittest.TestCase):
    def test_register_outcome_rejects_non_numeric(self):
        rs = RiskState()
        with self.assertRaises(ValueError):
            rs.register_trade_outcome("loss", "d1")
        with self.assertRaises(ValueError):
            rs.register_trade_outcome(True, "d1")


if __name__ == "__main__":
    unittest.main()
