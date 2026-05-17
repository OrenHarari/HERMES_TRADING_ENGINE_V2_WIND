"""Tests for hermes.backtest.report (Phase 3A)."""

import json
import os
import shutil
import tempfile
import unittest

from hermes.backtest.offline_runner import (
    BacktestConfig,
    run_offline_backtest,
)
from hermes.backtest.report import (
    PHASE3A_REPORT_KEYS,
    build_backtest_report,
    write_backtest_report,
)
from hermes.data.csv_loader import load_candles_csv
from hermes.decision.config import DecisionConfig

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "phase3a")


def _permissive_config(**overrides):
    defaults = dict(
        symbol="AMD", timeframe="1h",
        initial_equity=10_000.0,
        fee_pct=0.0, slippage_pct=0.0,
        take_profit_pct=0.02, stop_loss_pct=0.01,
        max_holding_bars=24, fixed_fraction=0.10,
        decision_config=DecisionConfig(
            min_confidence=0.0, min_agreement=0.0, allow_chop=True,
            volatility_min=0.0, volatility_max=1.0,
        ),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _run_amd_mode_a():
    loaded = load_candles_csv(
        os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv")
    )
    return run_offline_backtest(loaded["candles"], config=_permissive_config())


# ---- build_backtest_report --------------------------------------------

class TestBuildBacktestReport(unittest.TestCase):
    def test_all_phase3a_keys_present_at_top_level(self):
        result = _run_amd_mode_a()
        report = build_backtest_report(
            result, generated_at="2026-04-01T00:00:00+00:00",
            run_id="t1",
        )
        for k in PHASE3A_REPORT_KEYS:
            self.assertIn(k, report)

    def test_canonical_run_report_keys_preserved(self):
        result = _run_amd_mode_a()
        report = build_backtest_report(
            result, generated_at="2026-04-01T00:00:00+00:00",
        )
        for k in (
            "run_id", "generated_at", "system_mode",
            "decisions", "completed_trades",
            "performance_report", "learning_summary",
            "kill_switch_state", "summary", "notes",
        ):
            self.assertIn(k, report)

    def test_return_pct_formula(self):
        result = _run_amd_mode_a()
        report = build_backtest_report(
            result, generated_at="2026-04-01T00:00:00+00:00",
        )
        expected = (report["net_pnl"] / report["initial_equity"]) * 100.0
        self.assertAlmostEqual(report["return_pct"], expected, places=9)

    def test_symbol_and_timeframe_propagated(self):
        result = _run_amd_mode_a()
        report = build_backtest_report(
            result, generated_at="2026-04-01T00:00:00+00:00",
        )
        self.assertEqual(report["symbol"], "AMD")
        self.assertEqual(report["timeframe"], "1h")

    def test_equity_curve_preview_is_list_of_dicts(self):
        result = _run_amd_mode_a()
        report = build_backtest_report(
            result, generated_at="2026-04-01T00:00:00+00:00",
        )
        ec = report["equity_curve_preview"]
        self.assertIsInstance(ec, list)
        for point in ec:
            self.assertIn("timestamp", point)
            self.assertIn("equity", point)

    def test_completed_trades_preview_truncated(self):
        result = _run_amd_mode_a()
        report = build_backtest_report(
            result, generated_at="2026-04-01T00:00:00+00:00",
        )
        # 30-bar fixture won't produce >20 trades; just assert the cap.
        self.assertLessEqual(len(report["completed_trades_preview"]), 20)

    def test_no_trades_returns_zero_return_pct(self):
        # Force every decision to block.
        loaded = load_candles_csv(
            os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv")
        )
        cfg = _permissive_config(
            decision_config=DecisionConfig(
                min_confidence=1.0, min_agreement=1.0,
                allow_chop=True, volatility_min=0.0, volatility_max=1.0,
            )
        )
        result = run_offline_backtest(loaded["candles"], config=cfg)
        report = build_backtest_report(
            result, generated_at="2026-04-01T00:00:00+00:00",
        )
        self.assertEqual(report["trade_count"], 0)
        self.assertEqual(report["return_pct"], 0.0)
        self.assertEqual(report["net_pnl"], 0.0)

    def test_deterministic_with_explicit_generated_at(self):
        result = _run_amd_mode_a()
        a = build_backtest_report(
            result, generated_at="2026-04-01T00:00:00+00:00", run_id="r1"
        )
        b = build_backtest_report(
            result, generated_at="2026-04-01T00:00:00+00:00", run_id="r1"
        )
        self.assertEqual(a, b)

    def test_rejects_none_result(self):
        with self.assertRaises(ValueError):
            build_backtest_report(None)


# ---- write_backtest_report --------------------------------------------

class TestWriteBacktestReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_p3a_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_both_files(self):
        result = _run_amd_mode_a()
        out = write_backtest_report(
            result, output_dir=self.tmp,
            generated_at="2026-04-01T00:00:00+00:00",
            run_id="phase3a-amd-test",
        )
        self.assertTrue(os.path.exists(out["json_path"]))
        self.assertTrue(os.path.exists(out["html_path"]))

    def test_json_round_trips_with_phase3a_keys(self):
        result = _run_amd_mode_a()
        out = write_backtest_report(
            result, output_dir=self.tmp,
            generated_at="2026-04-01T00:00:00+00:00",
            run_id="phase3a-amd-test",
        )
        with open(out["json_path"], "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        for k in PHASE3A_REPORT_KEYS:
            self.assertIn(k, loaded)
        self.assertEqual(loaded["symbol"], "AMD")

    def test_html_starts_with_doctype(self):
        result = _run_amd_mode_a()
        out = write_backtest_report(
            result, output_dir=self.tmp,
            generated_at="2026-04-01T00:00:00+00:00",
        )
        with open(out["html_path"], "r", encoding="utf-8") as fh:
            html = fh.read()
        self.assertTrue(html.startswith("<!DOCTYPE html>"))


if __name__ == "__main__":
    unittest.main()
