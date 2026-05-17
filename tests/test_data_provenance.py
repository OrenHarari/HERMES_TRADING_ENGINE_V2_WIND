"""Tests for the data-provenance rule (Phase 3A addendum).

Covers:
  1. build_data_provenance() shape + validation.
  2. Synthetic test-fixture provenance (is_synthetic=True, fixture warning).
  3. User-provided CSV provenance (is_synthetic=False, user warning).
  4. Report builder injects data_provenance into JSON top-level.
  5. Report writer's HTML contains the warning text near the top.
  6. Default behavior when no provenance is passed: user_provided fallback,
     never silently synthetic.
  7. CLI --source=test_fixture_synthetic produces synthetic block.
  8. CLI default (no --source) produces user_provided block.
"""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest

from hermes.backtest.offline_runner import BacktestConfig, run_offline_backtest
from hermes.backtest.report import build_backtest_report, write_backtest_report
from hermes.data.csv_loader import load_candles_csv
from hermes.data.provenance import (
    DATA_PROVENANCE_KEYS,
    DATA_SOURCE_TEST_FIXTURE,
    DATA_SOURCE_USER_PROVIDED,
    VALID_SOURCES,
    WARNING_TEST_FIXTURE,
    WARNING_USER_PROVIDED,
    build_data_provenance,
)
from hermes.decision.config import DecisionConfig

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


def _permissive_config():
    return BacktestConfig(
        symbol="AMD", timeframe="1h",
        initial_equity=10_000.0, fee_pct=0.0, slippage_pct=0.0,
        take_profit_pct=0.02, stop_loss_pct=0.01,
        max_holding_bars=24, fixed_fraction=0.10,
        decision_config=DecisionConfig(
            min_confidence=0.0, min_agreement=0.0, allow_chop=True,
            volatility_min=0.0, volatility_max=1.0,
        ),
    )


def _run_amd_mode_a():
    loaded = load_candles_csv(
        os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv")
    )
    return run_offline_backtest(
        loaded["candles"], config=_permissive_config()
    ), loaded


# =========================================================================
# build_data_provenance
# =========================================================================

class TestBuildDataProvenance(unittest.TestCase):
    def test_synthetic_fixture(self):
        p = build_data_provenance(
            file_path="tests/fixtures/phase3a/sample.csv",
            symbol="AMD", timeframe="1h",
            row_count=30,
            start_timestamp=1700000000, end_timestamp=1700100000,
            source=DATA_SOURCE_TEST_FIXTURE,
        )
        self.assertTrue(p["is_synthetic"])
        self.assertEqual(p["source"], "test_fixture_synthetic")
        self.assertEqual(p["warning"], WARNING_TEST_FIXTURE)

    def test_user_provided(self):
        p = build_data_provenance(
            file_path="C:/data/AMD_1h.csv",
            symbol="AMD", timeframe="1h",
            row_count=2160,
            start_timestamp=1700000000, end_timestamp=1707776000,
            source=DATA_SOURCE_USER_PROVIDED,
        )
        self.assertFalse(p["is_synthetic"])
        self.assertEqual(p["source"], "user_provided")
        self.assertEqual(p["warning"], WARNING_USER_PROVIDED)

    def test_default_source_is_user_provided(self):
        p = build_data_provenance(
            file_path="x", symbol="AMD", timeframe="1h",
            row_count=10, start_timestamp=1, end_timestamp=2,
        )
        self.assertEqual(p["source"], DATA_SOURCE_USER_PROVIDED)
        self.assertFalse(p["is_synthetic"])

    def test_canonical_key_set(self):
        p = build_data_provenance(
            file_path="x", symbol="AMD", timeframe="1h",
            row_count=10, start_timestamp=1, end_timestamp=2,
            source=DATA_SOURCE_TEST_FIXTURE,
        )
        self.assertEqual(set(p.keys()), set(DATA_PROVENANCE_KEYS))

    def test_invalid_source_rejected(self):
        with self.assertRaises(ValueError):
            build_data_provenance(
                file_path="x", symbol="AMD", timeframe="1h",
                row_count=10, start_timestamp=1, end_timestamp=2,
                source="downloaded_from_yahoo",
            )

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            build_data_provenance(
                file_path="x", symbol="", timeframe="1h",
                row_count=10, start_timestamp=1, end_timestamp=2,
                source=DATA_SOURCE_USER_PROVIDED,
            )
        with self.assertRaises(ValueError):
            build_data_provenance(
                file_path="x", symbol="AMD", timeframe="1h",
                row_count=0, start_timestamp=1, end_timestamp=2,
                source=DATA_SOURCE_USER_PROVIDED,
            )
        with self.assertRaises(ValueError):
            build_data_provenance(
                file_path="x", symbol="AMD", timeframe="1h",
                row_count=10, start_timestamp=10, end_timestamp=5,
                source=DATA_SOURCE_USER_PROVIDED,
            )

    def test_valid_sources_constant(self):
        self.assertEqual(
            set(VALID_SOURCES),
            {"user_provided", "test_fixture_synthetic"},
        )


# =========================================================================
# Report builder integration
# =========================================================================

class TestReportProvenanceIntegration(unittest.TestCase):
    def test_synthetic_block_appears_in_json(self):
        result, loaded = _run_amd_mode_a()
        prov = build_data_provenance(
            file_path=os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv"),
            symbol="AMD", timeframe="1h",
            row_count=loaded["row_count"],
            start_timestamp=int(loaded["first_timestamp"]),
            end_timestamp=int(loaded["last_timestamp"]),
            source=DATA_SOURCE_TEST_FIXTURE,
        )
        report = build_backtest_report(
            result,
            generated_at="2026-04-27T00:00:00+00:00",
            data_provenance=prov,
        )
        self.assertIn("data_provenance", report)
        self.assertEqual(set(report["data_provenance"].keys()),
                         set(DATA_PROVENANCE_KEYS))
        self.assertTrue(report["data_provenance"]["is_synthetic"])
        self.assertEqual(report["data_provenance"]["warning"],
                         WARNING_TEST_FIXTURE)

    def test_user_provided_block_appears_in_json(self):
        result, loaded = _run_amd_mode_a()
        prov = build_data_provenance(
            file_path="/some/local/AMD_1h.csv",
            symbol="AMD", timeframe="1h",
            row_count=loaded["row_count"],
            start_timestamp=int(loaded["first_timestamp"]),
            end_timestamp=int(loaded["last_timestamp"]),
            source=DATA_SOURCE_USER_PROVIDED,
        )
        report = build_backtest_report(
            result,
            generated_at="2026-04-27T00:00:00+00:00",
            data_provenance=prov,
        )
        self.assertIn("data_provenance", report)
        self.assertFalse(report["data_provenance"]["is_synthetic"])
        self.assertEqual(report["data_provenance"]["warning"],
                         WARNING_USER_PROVIDED)

    def test_default_provenance_falls_back_to_user_provided(self):
        # When a caller forgets data_provenance, the report MUST still
        # carry a block, and that block must NEVER claim synthetic.
        result, _ = _run_amd_mode_a()
        report = build_backtest_report(
            result,
            generated_at="2026-04-27T00:00:00+00:00",
            data_provenance=None,
        )
        prov = report["data_provenance"]
        self.assertEqual(prov["source"], DATA_SOURCE_USER_PROVIDED)
        self.assertFalse(prov["is_synthetic"])
        self.assertEqual(prov["warning"], WARNING_USER_PROVIDED)


# =========================================================================
# HTML banner injection
# =========================================================================

class TestHtmlBanner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_prov_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read_html(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def test_synthetic_banner_in_html(self):
        result, loaded = _run_amd_mode_a()
        prov = build_data_provenance(
            file_path=os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv"),
            symbol="AMD", timeframe="1h",
            row_count=loaded["row_count"],
            start_timestamp=int(loaded["first_timestamp"]),
            end_timestamp=int(loaded["last_timestamp"]),
            source=DATA_SOURCE_TEST_FIXTURE,
        )
        out = write_backtest_report(
            result, output_dir=self.tmp,
            generated_at="2026-04-27T00:00:00+00:00",
            data_provenance=prov,
        )
        html = self._read_html(out["html_path"])
        self.assertIn(WARNING_TEST_FIXTURE, html)
        self.assertIn("SYNTHETIC FIXTURE", html)
        self.assertIn("data-provenance", html)

    def test_user_provided_banner_in_html(self):
        result, loaded = _run_amd_mode_a()
        prov = build_data_provenance(
            file_path="/some/local/AMD_1h.csv",
            symbol="AMD", timeframe="1h",
            row_count=loaded["row_count"],
            start_timestamp=int(loaded["first_timestamp"]),
            end_timestamp=int(loaded["last_timestamp"]),
            source=DATA_SOURCE_USER_PROVIDED,
        )
        out = write_backtest_report(
            result, output_dir=self.tmp,
            generated_at="2026-04-27T00:00:00+00:00",
            data_provenance=prov,
        )
        html = self._read_html(out["html_path"])
        self.assertIn(WARNING_USER_PROVIDED, html)
        self.assertIn("USER-PROVIDED CSV", html)

    def test_banner_appears_before_summary_section(self):
        result, _ = _run_amd_mode_a()
        out = write_backtest_report(
            result, output_dir=self.tmp,
            generated_at="2026-04-27T00:00:00+00:00",
        )
        html = self._read_html(out["html_path"])
        banner_idx = html.find("data-provenance")
        summary_idx = html.find("<h2>Summary</h2>")
        self.assertNotEqual(banner_idx, -1)
        self.assertNotEqual(summary_idx, -1)
        self.assertLess(banner_idx, summary_idx)

    def test_banner_html_escapes_file_path(self):
        # Defensive: a malicious file_path with HTML must not break the page.
        result, loaded = _run_amd_mode_a()
        prov = build_data_provenance(
            file_path="<script>alert(1)</script>",
            symbol="AMD", timeframe="1h",
            row_count=loaded["row_count"],
            start_timestamp=int(loaded["first_timestamp"]),
            end_timestamp=int(loaded["last_timestamp"]),
            source=DATA_SOURCE_USER_PROVIDED,
        )
        out = write_backtest_report(
            result, output_dir=self.tmp,
            generated_at="2026-04-27T00:00:00+00:00",
            data_provenance=prov,
        )
        html = self._read_html(out["html_path"])
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)


# =========================================================================
# CLI integration
# =========================================================================

class TestCliProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_cli_prov_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read_json(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def test_cli_with_synthetic_source_produces_synthetic_block(self):
        cli = _load_cli_module()
        rc = cli.main([
            "--csv", os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv"),
            "--source", DATA_SOURCE_TEST_FIXTURE,
            "--output-dir", self.tmp,
        ])
        self.assertEqual(rc, 0)
        report = self._read_json(os.path.join(self.tmp, "latest_run.json"))
        self.assertEqual(
            report["data_provenance"]["source"], DATA_SOURCE_TEST_FIXTURE
        )
        self.assertTrue(report["data_provenance"]["is_synthetic"])
        self.assertEqual(
            report["data_provenance"]["warning"], WARNING_TEST_FIXTURE
        )

    def test_cli_default_source_is_user_provided(self):
        cli = _load_cli_module()
        rc = cli.main([
            "--csv", os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv"),
            "--output-dir", self.tmp,
        ])
        self.assertEqual(rc, 0)
        report = self._read_json(os.path.join(self.tmp, "latest_run.json"))
        self.assertEqual(
            report["data_provenance"]["source"], DATA_SOURCE_USER_PROVIDED
        )
        self.assertFalse(report["data_provenance"]["is_synthetic"])
        self.assertEqual(
            report["data_provenance"]["warning"], WARNING_USER_PROVIDED
        )

    def test_cli_rejects_invalid_source(self):
        cli = _load_cli_module()
        with self.assertRaises(SystemExit):
            cli.main([
                "--csv", os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv"),
                "--source", "downloaded_from_yahoo",
                "--output-dir", self.tmp,
            ])


if __name__ == "__main__":
    unittest.main()
