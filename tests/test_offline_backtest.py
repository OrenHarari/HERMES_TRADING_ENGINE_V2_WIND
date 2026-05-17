"""Tests for hermes.backtest.offline_runner (Phase 3A)."""

import os
import unittest

from hermes.backtest.offline_runner import (
    EXIT_END_OF_DATA,
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT,
    EXIT_TIME_STOP,
    BacktestConfig,
    OfflineBacktestResult,
    run_offline_backtest,
)
from hermes.data.csv_loader import load_candles_csv
from hermes.decision.config import DecisionConfig

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "phase3a")


# ---- helpers -----------------------------------------------------------

def _permissive_config(**overrides):
    """Phase-3A test config with thresholds lowered so trades reliably
    open during exit-rule tests. Phase-1 defaults remain in production."""
    defaults = dict(
        symbol="AMD",
        timeframe="1h",
        initial_equity=10_000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        spread_pct=0.0,
        take_profit_pct=0.02,
        stop_loss_pct=0.01,
        max_holding_bars=24,
        fixed_fraction=0.10,
        decision_config=DecisionConfig(
            min_confidence=0.0,
            min_agreement=0.0,
            allow_chop=True,
            volatility_min=0.0,
            volatility_max=1.0,
        ),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _max_signal_provider(candles, current_index):
    return {"sequence_value": 0.95, "amd_value": 0.95, "combined_value": 0.95}


def _zero_signal_provider(candles, current_index):
    return {"sequence_value": 0.0, "amd_value": 0.0, "combined_value": 0.0}


def _build_candles(prices, start_ts=1700000000, ts_step=3600):
    """Build candles from a flat list of close prices.
    Each bar's OHLC is set to that close +/- 0.05 for volume of 1500.
    Use _build_candles_full for custom OHLC."""
    out = []
    for i, p in enumerate(prices):
        out.append({
            "timestamp": start_ts + i * ts_step,
            "open": p, "high": p + 0.05, "low": p - 0.05,
            "close": p, "volume": 1500.0,
        })
    return out


def _build_candles_full(rows, start_ts=1700000000, ts_step=3600):
    """rows: list of (open, high, low, close)."""
    out = []
    for i, (o, h, low, c) in enumerate(rows):
        out.append({
            "timestamp": start_ts + i * ts_step,
            "open": o, "high": h, "low": low, "close": c,
            "volume": 1500.0,
        })
    return out


def _warmup_uptrend(n=20, start=100.0, step=0.10):
    """Smooth uptrend with mild bar range (no exits triggered)."""
    rows = []
    for i in range(n):
        c = start + i * step
        rows.append((c, c + 0.05, c - 0.05, c))
    return rows


# ---- BacktestConfig validation ----------------------------------------

class TestBacktestConfigValidation(unittest.TestCase):
    def test_defaults_construct(self):
        cfg = BacktestConfig()
        self.assertEqual(cfg.symbol, "AMD")
        self.assertEqual(cfg.timeframe, "1h")
        self.assertEqual(cfg.fixed_fraction, 0.10)

    def test_invalid_initial_equity(self):
        with self.assertRaises(ValueError):
            BacktestConfig(initial_equity=0)

    def test_invalid_fixed_fraction(self):
        with self.assertRaises(ValueError):
            BacktestConfig(fixed_fraction=0.0)
        with self.assertRaises(ValueError):
            BacktestConfig(fixed_fraction=1.5)

    def test_invalid_take_profit(self):
        with self.assertRaises(ValueError):
            BacktestConfig(take_profit_pct=0.0)

    def test_invalid_stop_loss(self):
        with self.assertRaises(ValueError):
            BacktestConfig(stop_loss_pct=0.0)


# ---- E2E happy paths --------------------------------------------------

class TestE2EHappyPaths(unittest.TestCase):
    def test_runs_on_mode_a_fixture(self):
        loaded = load_candles_csv(
            os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv")
        )
        result = run_offline_backtest(
            loaded["candles"], config=_permissive_config()
        )
        self.assertIsInstance(result, OfflineBacktestResult)
        self.assertEqual(result.candles_count, 30)
        self.assertEqual(result.start_timestamp, 1700000000)
        self.assertEqual(result.end_timestamp, 1700104400)
        # Decisions list length = candles - 1 (last bar can't open).
        self.assertEqual(len(result.decisions), 29)

    def test_runs_on_mode_b_fixture(self):
        loaded = load_candles_csv(
            os.path.join(FIXTURES, "AMD_1h_mode_b_ohlcv.csv")
        )
        result = run_offline_backtest(
            loaded["candles"], config=_permissive_config()
        )
        self.assertIsInstance(result, OfflineBacktestResult)
        self.assertTrue(result.cost_model_applied)


# ---- Exit rules -------------------------------------------------------

class TestExitRules(unittest.TestCase):
    def test_take_profit_exit(self):
        # Warmup uptrend, then a rally bar that breaches TP=2% above entry.
        rows = _warmup_uptrend(20, start=100.0, step=0.10)
        # Bar 20 = decision bar; bar 21 = entry bar (open at 102 + ...).
        # Set bar 21 entry near 102, then bar 22 high blasts above 104.
        # We'll let bar 21 open at the natural progression, then make bar 22
        # rally hard so its high >= entry*1.02.
        # Append more bars: bar 20 same warmup; bar 21 fills entry; bar 22 spike.
        rows.append((101.95, 102.00, 101.90, 102.00))  # bar 20
        rows.append((102.00, 102.10, 101.95, 102.05))  # bar 21 entry @ 102.00
        rows.append((102.05, 105.00, 102.05, 104.50))  # bar 22 RALLY (high=105)
        rows += [(104.5, 104.6, 104.4, 104.5)] * 5
        candles = _build_candles_full(rows)
        result = run_offline_backtest(
            candles,
            config=_permissive_config(
                take_profit_pct=0.02, stop_loss_pct=0.05
            ),
            signal_provider=_max_signal_provider,
        )
        self.assertGreaterEqual(len(result.completed_trades), 1)
        first_trade = result.completed_trades[0]
        self.assertEqual(first_trade["exit_reason"], EXIT_TAKE_PROFIT)
        self.assertGreater(first_trade["net_pnl"], 0.0)

    def test_stop_loss_exit(self):
        # Warmup, then a CRASH bar that breaches SL=1% below entry.
        rows = _warmup_uptrend(20, start=100.0, step=0.10)
        rows.append((101.95, 102.00, 101.90, 102.00))  # bar 20
        rows.append((102.00, 102.10, 101.95, 102.05))  # bar 21 entry @ 102.00
        rows.append((102.05, 102.05, 99.00, 99.50))    # bar 22 CRASH
        rows += [(99.5, 99.6, 99.4, 99.5)] * 5
        candles = _build_candles_full(rows)
        result = run_offline_backtest(
            candles,
            config=_permissive_config(
                take_profit_pct=0.05, stop_loss_pct=0.01
            ),
            signal_provider=_max_signal_provider,
        )
        self.assertGreaterEqual(len(result.completed_trades), 1)
        first_trade = result.completed_trades[0]
        self.assertEqual(first_trade["exit_reason"], EXIT_STOP_LOSS)
        self.assertLess(first_trade["net_pnl"], 0.0)

    def test_stop_wins_within_bar_tie(self):
        # Bar that hits BOTH stop and tp simultaneously -> stop wins.
        rows = _warmup_uptrend(20, start=100.0, step=0.10)
        rows.append((101.95, 102.00, 101.90, 102.00))  # bar 20
        rows.append((102.00, 102.10, 101.95, 102.05))  # bar 21 entry @ 102.00
        # bar 22: high blasts above tp AND low pierces stop -> stop wins.
        rows.append((102.05, 110.00, 95.00, 102.00))
        rows += [(102.0, 102.1, 101.9, 102.0)] * 5
        candles = _build_candles_full(rows)
        result = run_offline_backtest(
            candles,
            config=_permissive_config(
                take_profit_pct=0.02, stop_loss_pct=0.01
            ),
            signal_provider=_max_signal_provider,
        )
        self.assertEqual(result.completed_trades[0]["exit_reason"], EXIT_STOP_LOSS)

    def test_time_stop_exit(self):
        # Flat after entry, no SL/TP hit -> time stop.
        rows = _warmup_uptrend(20, start=100.0, step=0.10)
        rows.append((101.95, 102.00, 101.90, 102.00))  # bar 20
        rows.append((102.00, 102.05, 101.95, 102.00))  # bar 21 entry
        # 5 bars where price stays inside SL/TP band; max_holding_bars=3.
        rows += [(102.00, 102.05, 101.95, 102.00)] * 6
        candles = _build_candles_full(rows)
        result = run_offline_backtest(
            candles,
            config=_permissive_config(
                take_profit_pct=0.05, stop_loss_pct=0.05,
                max_holding_bars=3,
            ),
            signal_provider=_max_signal_provider,
        )
        self.assertEqual(result.completed_trades[0]["exit_reason"], EXIT_TIME_STOP)

    def test_end_of_data_exit(self):
        # Run ends with position still open and no SL/TP hit, no time stop.
        rows = _warmup_uptrend(20, start=100.0, step=0.10)
        rows.append((101.95, 102.00, 101.90, 102.00))  # bar 20 decide
        rows.append((102.00, 102.05, 101.95, 102.00))  # bar 21 entry
        rows.append((102.00, 102.05, 101.95, 102.01))  # bar 22 final - no exit
        candles = _build_candles_full(rows)
        result = run_offline_backtest(
            candles,
            config=_permissive_config(
                take_profit_pct=0.05, stop_loss_pct=0.05,
                max_holding_bars=99,
            ),
            signal_provider=_max_signal_provider,
        )
        self.assertEqual(
            result.completed_trades[0]["exit_reason"], EXIT_END_OF_DATA
        )


# ---- Invariants -------------------------------------------------------

class TestInvariants(unittest.TestCase):
    def test_one_position_at_a_time(self):
        # Long flat sequence with high signal: only one trade at a time.
        rows = _warmup_uptrend(20, start=100.0, step=0.10)
        rows += [(102.0, 102.05, 101.95, 102.0)] * 50
        candles = _build_candles_full(rows)
        result = run_offline_backtest(
            candles,
            config=_permissive_config(max_holding_bars=5),
            signal_provider=_max_signal_provider,
        )
        # Verify entry/exit timestamps never overlap.
        for i in range(1, len(result.completed_trades)):
            prev = result.completed_trades[i - 1]
            curr = result.completed_trades[i]
            self.assertGreater(
                curr["entry_timestamp"], prev["exit_timestamp"],
                msg="positions overlapped in time"
            )
        # And blocked_reasons may include 'already_in_position'.
        # (Not asserted as required, since timing makes it probabilistic.)

    def test_deterministic(self):
        loaded = load_candles_csv(
            os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv")
        )
        cfg = _permissive_config()
        a = run_offline_backtest(loaded["candles"], config=cfg)
        b = run_offline_backtest(loaded["candles"], config=cfg)
        self.assertEqual(a.net_pnl, b.net_pnl)
        self.assertEqual(a.return_pct, b.return_pct)
        self.assertEqual(len(a.completed_trades), len(b.completed_trades))
        self.assertEqual(a.equity_curve, b.equity_curve)

    def test_no_future_data_used(self):
        loaded = load_candles_csv(
            os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv")
        )
        cfg = _permissive_config()
        a = run_offline_backtest(loaded["candles"], config=cfg)
        # Mutate the LAST candle and verify only the very-final-bar exit
        # is affected (decisions made on earlier bars should be unchanged).
        candles2 = [dict(c) for c in loaded["candles"]]
        candles2[-1]["close"] = 1.0  # absurd value
        candles2[-1]["high"] = 1.0
        candles2[-1]["low"] = 1.0
        b = run_offline_backtest(candles2, config=cfg)
        # All decisions except the last (made at bar n-2 reading bars 0..n-2)
        # are unaffected -- they don't read bar n-1.
        self.assertEqual(len(a.decisions), len(b.decisions))
        for i in range(len(a.decisions) - 1):
            self.assertEqual(
                {k: a.decisions[i][k] for k in ("trade_allowed", "reason",
                                                "regime", "confidence")},
                {k: b.decisions[i][k] for k in ("trade_allowed", "reason",
                                                "regime", "confidence")},
            )

    def test_return_pct_formula(self):
        result = run_offline_backtest(
            _build_candles([100.0 + i * 0.1 for i in range(30)]),
            config=_permissive_config(initial_equity=50_000.0),
        )
        expected = (result.net_pnl / 50_000.0) * 100.0
        self.assertAlmostEqual(result.return_pct, expected, places=9)

    def test_final_equity_equals_initial_plus_net_pnl(self):
        result = run_offline_backtest(
            _build_candles([100.0 + i * 0.1 for i in range(30)]),
            config=_permissive_config(initial_equity=10_000.0),
        )
        self.assertAlmostEqual(
            result.final_equity,
            result.initial_equity + result.net_pnl,
            places=6,
        )

    def test_max_drawdown_non_negative(self):
        result = run_offline_backtest(
            _build_candles([100.0 + i * 0.1 for i in range(30)]),
            config=_permissive_config(),
        )
        self.assertGreaterEqual(result.max_drawdown, 0.0)
        self.assertGreaterEqual(result.max_drawdown_pct, 0.0)

    def test_decisions_count_equals_candles_minus_one(self):
        result = run_offline_backtest(
            _build_candles([100.0 + i * 0.1 for i in range(30)]),
            config=_permissive_config(),
        )
        self.assertEqual(len(result.decisions), 29)

    def test_cost_model_applied_flag(self):
        result = run_offline_backtest(
            _build_candles([100.0 + i * 0.1 for i in range(30)]),
            config=_permissive_config(),
        )
        self.assertTrue(result.cost_model_applied)


# ---- Cost model integration (no duplicated math) ----------------------

class TestCostModelIntegration(unittest.TestCase):
    def test_zero_fees_gross_equals_net(self):
        rows = _warmup_uptrend(20, start=100.0, step=0.10)
        rows.append((101.95, 102.00, 101.90, 102.00))
        rows.append((102.00, 102.10, 101.95, 102.05))  # entry @ 102.00
        rows.append((102.05, 105.00, 102.05, 104.50))  # TP rally
        rows += [(104.5, 104.6, 104.4, 104.5)] * 3
        candles = _build_candles_full(rows)
        result = run_offline_backtest(
            candles,
            config=_permissive_config(fee_pct=0.0, slippage_pct=0.0),
            signal_provider=_max_signal_provider,
        )
        t = result.completed_trades[0]
        self.assertAlmostEqual(t["gross_pnl"], t["net_pnl"], places=9)
        self.assertEqual(t["fees"], 0.0)
        self.assertEqual(t["slippage"], 0.0)

    def test_positive_fees_reduce_net(self):
        rows = _warmup_uptrend(20, start=100.0, step=0.10)
        rows.append((101.95, 102.00, 101.90, 102.00))
        rows.append((102.00, 102.10, 101.95, 102.05))
        rows.append((102.05, 105.00, 102.05, 104.50))
        rows += [(104.5, 104.6, 104.4, 104.5)] * 3
        candles = _build_candles_full(rows)
        result = run_offline_backtest(
            candles,
            config=_permissive_config(fee_pct=0.001, slippage_pct=0.001),
            signal_provider=_max_signal_provider,
        )
        t = result.completed_trades[0]
        self.assertLess(t["net_pnl"], t["gross_pnl"])
        self.assertGreater(t["fees"], 0.0)
        self.assertGreater(t["slippage"], 0.0)


# ---- Sizing rules: fixed-fraction is the ONLY simulated sizing -------

class TestSizingRules(unittest.TestCase):
    def test_fixed_fraction_drives_notional_not_core(self):
        # Build a setup that opens exactly one trade with predictable entry.
        rows = _warmup_uptrend(20, start=100.0, step=0.10)
        rows.append((101.95, 102.00, 101.90, 102.00))
        rows.append((102.00, 102.10, 101.95, 102.05))  # entry @ 102.00
        rows.append((102.05, 105.00, 102.05, 104.50))  # exit
        rows += [(104.5, 104.6, 104.4, 104.5)] * 3
        candles = _build_candles_full(rows)
        result = run_offline_backtest(
            candles,
            config=_permissive_config(
                initial_equity=10_000.0, fixed_fraction=0.20,
                fee_pct=0.0, slippage_pct=0.0,
            ),
            signal_provider=_max_signal_provider,
        )
        t = result.completed_trades[0]
        expected_notional = 10_000.0 * 0.20
        self.assertAlmostEqual(t["notional"], expected_notional, places=6)
        # core_position_size is recorded (audit) and is 1.0 with default
        # DecisionConfig.base_position_size, NOT a dollar figure.
        self.assertIn("core_position_size", t)
        self.assertNotEqual(t["core_position_size"], expected_notional)
        # shares == notional / entry_price
        self.assertAlmostEqual(
            t["shares"], expected_notional / t["entry_price"], places=9
        )


# ---- Blocked reasons --------------------------------------------------

class TestBlockedReasons(unittest.TestCase):
    def test_blocked_reasons_count_populated_when_min_confidence_unreachable(self):
        # Force every decision to block by setting min_confidence above 1.0
        # ceiling -- guaranteed unreachable regardless of signal strength.
        candles = _build_candles([100.0 + i * 0.1 for i in range(30)])
        result = run_offline_backtest(
            candles,
            config=_permissive_config(
                decision_config=DecisionConfig(
                    min_confidence=1.0,
                    min_agreement=1.0,
                    allow_chop=True,
                    volatility_min=0.0,
                    volatility_max=1.0,
                ),
            ),
            signal_provider=_zero_signal_provider,
        )
        self.assertEqual(len(result.completed_trades), 0)
        self.assertGreater(sum(result.blocked_reasons_count.values()), 0)
        self.assertEqual(result.net_pnl, 0.0)
        self.assertEqual(result.return_pct, 0.0)


# ---- Validation guards -----------------------------------------------

class TestValidationGuards(unittest.TestCase):
    def test_too_few_candles_rejected(self):
        with self.assertRaises(ValueError):
            run_offline_backtest([], config=_permissive_config())
        with self.assertRaises(ValueError):
            run_offline_backtest(
                [{"timestamp": 1, "open": 1, "high": 1, "low": 1,
                  "close": 1, "volume": 1}],
                config=_permissive_config(),
            )

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            run_offline_backtest(
                _build_candles([100.0] * 5),
                config="not a config",
            )


if __name__ == "__main__":
    unittest.main()
