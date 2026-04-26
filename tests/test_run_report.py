"""Tests for the run-report layer."""

import json
import os
import shutil
import tempfile
import unittest

from hermes.reports.run_report import (
    DEFAULT_OUTPUT_DIR,
    REPORT_HTML_FILENAME,
    REPORT_JSON_FILENAME,
    REQUIRED_REPORT_KEYS,
    build_run_report,
    render_html,
    write_run_report,
)


def _decision(allowed, conf=0.7, regime="trend_up", size=10.0, reason=""):
    return {
        "trade_allowed": allowed,
        "confidence": conf,
        "agreement": 0.8,
        "regime": regime,
        "position_size": size,
        "reason": reason,
    }


def _trade(outcome="win", net_pnl=12.0):
    return {
        "outcome": outcome,
        "regime": "trend_up",
        "entry_price": 100.0,
        "exit_price": 110.0,
        "net_pnl": net_pnl,
        "exit_reason": "take_profit",
    }


def _perf(net_pnl=100.0, win_rate=0.6, trade_count=10):
    return {
        "net_pnl": net_pnl,
        "win_rate": win_rate,
        "avg_win": 12.0,
        "avg_loss": -5.0,
        "profit_factor": 2.4,
        "max_drawdown": 30.0,
        "trade_count": trade_count,
        "trades_per_regime": {"trend_up": trade_count},
        "stability_score": 0.5,
        "cost_model_applied": True,
    }


# ---- build_run_report ---------------------------------------------------

class TestBuildRunReport(unittest.TestCase):
    def test_required_keys_present(self):
        rep = build_run_report(generated_at="2025-01-01T00:00:00+00:00")
        for k in REQUIRED_REPORT_KEYS:
            self.assertIn(k, rep)

    def test_required_keys_match_spec(self):
        self.assertEqual(
            set(REQUIRED_REPORT_KEYS),
            {
                "run_id", "generated_at", "system_mode", "decisions",
                "completed_trades", "performance_report", "learning_summary",
                "kill_switch_state", "summary", "notes",
            },
        )

    def test_empty_inputs_yield_empty_collections(self):
        rep = build_run_report(generated_at="2025-01-01T00:00:00+00:00")
        self.assertEqual(rep["decisions"], [])
        self.assertEqual(rep["completed_trades"], [])
        self.assertEqual(rep["performance_report"], {})
        self.assertEqual(rep["learning_summary"], {})
        self.assertEqual(rep["kill_switch_state"], {})
        self.assertEqual(rep["notes"], "")

    def test_run_id_defaults_to_generated_at(self):
        rep = build_run_report(generated_at="2025-01-01T00:00:00+00:00")
        self.assertEqual(rep["run_id"], "2025-01-01T00:00:00+00:00")

    def test_explicit_run_id_used(self):
        rep = build_run_report(
            run_id="run-42", generated_at="2025-01-01T00:00:00+00:00"
        )
        self.assertEqual(rep["run_id"], "run-42")

    def test_summary_counts_correct(self):
        rep = build_run_report(
            decisions=[_decision(True), _decision(False), _decision(True)],
            completed_trades=[_trade(), _trade(), _trade(), _trade()],
            generated_at="2025-01-01T00:00:00+00:00",
        )
        self.assertEqual(rep["summary"]["total_decisions"], 3)
        self.assertEqual(rep["summary"]["approved_decisions"], 2)
        self.assertEqual(rep["summary"]["blocked_decisions"], 1)
        self.assertEqual(rep["summary"]["total_trades"], 4)

    def test_default_generated_at_is_iso_string(self):
        rep = build_run_report()
        self.assertIsInstance(rep["generated_at"], str)
        # ISO 8601 UTC should contain a 'T' and end with '+00:00'.
        self.assertIn("T", rep["generated_at"])

    def test_inputs_are_copied_not_aliased(self):
        decisions = [_decision(True)]
        rep = build_run_report(
            decisions=decisions, generated_at="2025-01-01T00:00:00+00:00"
        )
        decisions.append(_decision(False))
        # Original list mutated; report's list should be unaffected (build
        # function snapshots via list()).
        self.assertEqual(len(rep["decisions"]), 1)

    def test_dict_inputs_are_copied(self):
        perf = _perf()
        rep = build_run_report(
            performance_report=perf,
            generated_at="2025-01-01T00:00:00+00:00",
        )
        perf["net_pnl"] = -999.0
        self.assertNotEqual(rep["performance_report"]["net_pnl"], -999.0)

    def test_deterministic_with_explicit_generated_at(self):
        a = build_run_report(
            decisions=[_decision(True)],
            generated_at="2025-01-01T00:00:00+00:00",
            run_id="r1",
        )
        b = build_run_report(
            decisions=[_decision(True)],
            generated_at="2025-01-01T00:00:00+00:00",
            run_id="r1",
        )
        self.assertEqual(a, b)


# ---- render_html ---------------------------------------------------------

class TestRenderHtml(unittest.TestCase):
    def test_renders_a_string(self):
        rep = build_run_report(generated_at="2025-01-01T00:00:00+00:00")
        out = render_html(rep)
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("<!DOCTYPE html>"))
        self.assertIn("</html>", out)

    def test_contains_run_id_and_generated_at(self):
        rep = build_run_report(
            run_id="my-run-7", generated_at="2025-04-01T10:00:00+00:00"
        )
        out = render_html(rep)
        self.assertIn("my-run-7", out)
        self.assertIn("2025-04-01T10:00:00+00:00", out)

    def test_escapes_html_in_run_id(self):
        rep = build_run_report(
            run_id="<script>alert(1)</script>",
            generated_at="2025-01-01T00:00:00+00:00",
        )
        out = render_html(rep)
        self.assertNotIn("<script>alert(1)</script>", out)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", out)

    def test_escapes_html_in_decision_reason(self):
        rep = build_run_report(
            decisions=[_decision(False, reason="<b>boom</b>")],
            generated_at="2025-01-01T00:00:00+00:00",
        )
        out = render_html(rep)
        self.assertNotIn("<b>boom</b>", out)
        self.assertIn("&lt;b&gt;boom&lt;/b&gt;", out)

    def test_empty_sections_show_placeholder(self):
        rep = build_run_report(generated_at="2025-01-01T00:00:00+00:00")
        out = render_html(rep)
        # Each empty section renders an italic empty placeholder.
        self.assertIn("(no decisions)", out)
        self.assertIn("(no completed trades)", out)
        self.assertIn("(no performance report)", out)
        self.assertIn("(no learning summary)", out)
        self.assertIn("(no kill switch state recorded)", out)

    def test_decisions_rendered_with_status(self):
        rep = build_run_report(
            decisions=[_decision(True), _decision(False, reason="low_confidence")],
            generated_at="2025-01-01T00:00:00+00:00",
        )
        out = render_html(rep)
        self.assertIn("PASS", out)
        self.assertIn("BLOCKED", out)
        self.assertIn("low_confidence", out)

    def test_performance_section_renders_metrics(self):
        rep = build_run_report(
            performance_report=_perf(),
            generated_at="2025-01-01T00:00:00+00:00",
        )
        out = render_html(rep)
        self.assertIn("net_pnl", out)
        self.assertIn("profit_factor", out)
        self.assertIn("trades_per_regime", out)

    def test_kill_switch_active_renders_badge(self):
        rep = build_run_report(
            kill_switch_state={
                "active": True, "reason": "stale_data_feed",
                "activated_ts": 1000, "log": [{"event": "kill_switch_activated"}],
            },
            generated_at="2025-01-01T00:00:00+00:00",
        )
        out = render_html(rep)
        self.assertIn("ACTIVE", out)
        self.assertIn("stale_data_feed", out)

    def test_kill_switch_inactive_renders_inactive_badge(self):
        rep = build_run_report(
            kill_switch_state={"active": False, "reason": "", "activated_ts": None,
                               "log": []},
            generated_at="2025-01-01T00:00:00+00:00",
        )
        out = render_html(rep)
        self.assertIn("INACTIVE", out)

    def test_no_javascript_in_output(self):
        rep = build_run_report(generated_at="2025-01-01T00:00:00+00:00")
        out = render_html(rep).lower()
        self.assertNotIn("<script", out)
        self.assertNotIn("javascript:", out)
        self.assertNotIn("onclick", out)
        self.assertNotIn("onload", out)

    def test_no_external_url_dependencies(self):
        rep = build_run_report(generated_at="2025-01-01T00:00:00+00:00")
        out = render_html(rep).lower()
        # No CDN-style URL references; everything is self-contained.
        self.assertNotIn("<link", out)
        self.assertNotIn("<script src", out)
        self.assertNotIn("//cdn.", out)
        self.assertNotIn("//unpkg.", out)

    def test_deterministic(self):
        rep = build_run_report(
            decisions=[_decision(True), _decision(False)],
            performance_report=_perf(),
            generated_at="2025-01-01T00:00:00+00:00",
            run_id="r1",
        )
        a = render_html(rep)
        b = render_html(rep)
        self.assertEqual(a, b)

    def test_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            render_html("not a dict")


# ---- write_run_report ---------------------------------------------------

class TestWriteRunReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_rep_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_both_files(self):
        rep = build_run_report(
            decisions=[_decision(True)],
            generated_at="2025-01-01T00:00:00+00:00",
        )
        out = write_run_report(rep, output_dir=self.tmp)
        self.assertTrue(os.path.exists(out["json_path"]))
        self.assertTrue(os.path.exists(out["html_path"]))
        self.assertTrue(out["json_path"].endswith(REPORT_JSON_FILENAME))
        self.assertTrue(out["html_path"].endswith(REPORT_HTML_FILENAME))

    def test_json_round_trips(self):
        rep = build_run_report(
            decisions=[_decision(True), _decision(False)],
            performance_report=_perf(),
            generated_at="2025-01-01T00:00:00+00:00",
            run_id="round-trip",
        )
        out = write_run_report(rep, output_dir=self.tmp)
        with open(out["json_path"], "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded, rep)

    def test_html_starts_with_doctype(self):
        rep = build_run_report(generated_at="2025-01-01T00:00:00+00:00")
        out = write_run_report(rep, output_dir=self.tmp)
        with open(out["html_path"], "r", encoding="utf-8") as fh:
            html = fh.read()
        self.assertTrue(html.startswith("<!DOCTYPE html>"))

    def test_creates_output_dir_if_missing(self):
        target = os.path.join(self.tmp, "nested", "reports")
        rep = build_run_report(generated_at="2025-01-01T00:00:00+00:00")
        out = write_run_report(rep, output_dir=target)
        self.assertTrue(os.path.isdir(target))
        self.assertTrue(os.path.exists(out["json_path"]))

    def test_overwrites_existing_files(self):
        rep1 = build_run_report(
            run_id="first", generated_at="2025-01-01T00:00:00+00:00",
        )
        rep2 = build_run_report(
            run_id="second", generated_at="2025-01-02T00:00:00+00:00",
        )
        write_run_report(rep1, output_dir=self.tmp)
        out = write_run_report(rep2, output_dir=self.tmp)
        with open(out["json_path"], "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["run_id"], "second")

    def test_default_output_dir_constant(self):
        self.assertEqual(DEFAULT_OUTPUT_DIR, "reports")

    def test_filenames_are_canonical(self):
        self.assertEqual(REPORT_JSON_FILENAME, "latest_run.json")
        self.assertEqual(REPORT_HTML_FILENAME, "latest_run.html")

    def test_rejects_non_dict_report(self):
        with self.assertRaises(ValueError):
            write_run_report("not a dict", output_dir=self.tmp)

    def test_rejects_empty_output_dir(self):
        with self.assertRaises(ValueError):
            write_run_report({"x": 1}, output_dir="")


if __name__ == "__main__":
    unittest.main()
