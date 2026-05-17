"""Smoke test for run_offline_backtest.py CLI (Phase 3A)."""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "phase3a")
CLI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "run_offline_backtest.py",
)


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_offline_backtest", CLI_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_cli_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cli_runs_on_amd_mode_a_fixture(self):
        cli = _load_cli_module()
        rc = cli.main([
            "--csv", os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv"),
            "--symbol", "AMD",
            "--timeframe", "1h",
            "--initial-equity", "10000",
            "--fee-pct", "0.0",
            "--slippage-pct", "0.0",
            "--take-profit", "0.02",
            "--stop-loss", "0.01",
            "--output-dir", self.tmp,
        ])
        self.assertEqual(rc, 0)
        json_path = os.path.join(self.tmp, "latest_run.json")
        html_path = os.path.join(self.tmp, "latest_run.html")
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(html_path))
        with open(json_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        self.assertEqual(report["symbol"], "AMD")
        self.assertEqual(report["timeframe"], "1h")
        self.assertIn("return_pct", report)
        self.assertIn("net_pnl", report)
        # return_pct formula property
        self.assertAlmostEqual(
            report["return_pct"],
            (report["net_pnl"] / report["initial_equity"]) * 100.0,
            places=9,
        )

    def test_cli_runs_on_amd_mode_b_fixture(self):
        cli = _load_cli_module()
        rc = cli.main([
            "--csv", os.path.join(FIXTURES, "AMD_1h_mode_b_ohlcv.csv"),
            "--initial-equity", "10000",
            "--fee-pct", "0.0",
            "--slippage-pct", "0.0",
            "--output-dir", self.tmp,
        ])
        self.assertEqual(rc, 0)
        json_path = os.path.join(self.tmp, "latest_run.json")
        with open(json_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        self.assertEqual(report["symbol"], "AMD")
        self.assertTrue(report["cost_model_applied"])

    def test_cli_rejects_missing_csv(self):
        cli = _load_cli_module()
        with self.assertRaises(SystemExit):
            cli.main([])  # --csv is required


if __name__ == "__main__":
    unittest.main()
