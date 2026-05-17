"""Phase 3A - Backtest report adapter.

Wraps an `OfflineBacktestResult` in the canonical run-report shape used
by `hermes.reports`, augmenting the top-level dict with the Phase-3A
required fields and the mandatory data_provenance block.

Key design choices:
  - Phase-1 / Phase-2 / pre-Phase-3A `hermes.reports.run_report` is NOT
    modified. We call its `build_run_report` and `render_html`, then
    splice the data-provenance banner into the rendered HTML in this
    file before writing.
  - `data_provenance` is REQUIRED in every report (defaulted to
    user_provided when not supplied). The default still produces a
    valid block -- never silently absent.
  - HTML is written atomically here (write-temp, fsync, replace).
"""

import html as _html
import os

from hermes.data.provenance import (
    DATA_SOURCE_USER_PROVIDED,
    build_data_provenance,
)
from hermes.decision.performance import compute_performance_report
from hermes.reports import build_run_report, render_html
from hermes.utils.json_io import write_json_atomic

REPORT_JSON_FILENAME = "latest_run.json"
REPORT_HTML_FILENAME = "latest_run.html"

# Maximum entries kept in the previews placed in the report so that
# JSON / HTML stay small for very long runs.
_MAX_TRADES_PREVIEW = 20
_MAX_EQUITY_PREVIEW = 100

PHASE3A_REPORT_KEYS = (
    "symbol", "timeframe",
    "start_timestamp", "end_timestamp",
    "initial_equity", "final_equity",
    "net_pnl", "return_pct",
    "max_drawdown", "max_drawdown_pct",
    "trade_count", "win_rate", "profit_factor",
    "avg_win", "avg_loss", "stability_score",
    "cost_model_applied",
    "blocked_reasons_count", "trades_per_regime",
    "completed_trades_preview", "equity_curve_preview",
    "data_provenance",
)


def _equity_curve_preview(equity_curve, max_points=_MAX_EQUITY_PREVIEW):
    if not equity_curve:
        return []
    n = len(equity_curve)
    if n <= max_points:
        return [dict(p) for p in equity_curve]
    half = max_points // 2
    head = equity_curve[:half]
    tail = equity_curve[-half:]
    return [dict(p) for p in head] + [dict(p) for p in tail]


def _default_provenance_for_result(result):
    """Build a default user_provided provenance block for a result.

    Used when the caller did NOT pass an explicit `data_provenance` --
    this keeps the report honest by NEVER claiming synthetic by default.
    """
    return build_data_provenance(
        file_path="",
        symbol=result.config.symbol,
        timeframe=result.config.timeframe,
        row_count=result.candles_count,
        start_timestamp=int(result.start_timestamp),
        end_timestamp=int(result.end_timestamp),
        source=DATA_SOURCE_USER_PROVIDED,
    )


def build_backtest_report(
    result, *,
    run_id=None,
    generated_at=None,
    notes=None,
    data_provenance=None,
):
    """Build a run-report dict for an OfflineBacktestResult.

    `data_provenance` is REQUIRED in the output. If the caller passes
    None, a default user_provided provenance block is built from
    `result` (with empty file_path).
    """
    if result is None:
        raise ValueError("result must not be None")

    if data_provenance is None:
        data_provenance = _default_provenance_for_result(result)
    elif not isinstance(data_provenance, dict):
        raise ValueError("data_provenance must be a dict or None")

    perf = compute_performance_report(
        result.completed_trades, cost_model_applied=result.cost_model_applied
    )

    base = build_run_report(
        decisions=result.decisions,
        completed_trades=result.completed_trades,
        performance_report=perf,
        learning_summary={},
        kill_switch_state={},
        system_mode="backtest_mode",
        run_id=run_id,
        notes=notes,
        generated_at=generated_at,
    )

    # ---- merge Phase-3A required top-level keys (additive) -------------
    base["symbol"] = result.config.symbol
    base["timeframe"] = result.config.timeframe
    base["start_timestamp"] = int(result.start_timestamp)
    base["end_timestamp"] = int(result.end_timestamp)
    base["initial_equity"] = float(result.initial_equity)
    base["final_equity"] = float(result.final_equity)
    base["net_pnl"] = float(result.net_pnl)
    base["return_pct"] = float(result.return_pct)
    base["max_drawdown"] = float(result.max_drawdown)
    base["max_drawdown_pct"] = float(result.max_drawdown_pct)
    base["trade_count"] = int(perf["trade_count"])
    base["win_rate"] = float(perf["win_rate"])
    base["profit_factor"] = (
        float(perf["profit_factor"])
        if perf["profit_factor"] != float("inf") else float("inf")
    )
    base["avg_win"] = float(perf["avg_win"])
    base["avg_loss"] = float(perf["avg_loss"])
    base["stability_score"] = float(perf["stability_score"])
    base["cost_model_applied"] = bool(perf["cost_model_applied"])
    base["blocked_reasons_count"] = dict(result.blocked_reasons_count)
    base["trades_per_regime"] = dict(perf["trades_per_regime"])
    base["completed_trades_preview"] = [
        dict(t) for t in result.completed_trades[:_MAX_TRADES_PREVIEW]
    ]
    base["equity_curve_preview"] = _equity_curve_preview(result.equity_curve)
    base["data_provenance"] = dict(data_provenance)
    return base


def _provenance_banner_html(provenance):
    """Render a high-visibility provenance banner near the top of the HTML.

    The banner is fully self-contained (inline styles), no JS, no
    external resources.
    """
    if not isinstance(provenance, dict):
        return ""
    is_synth = bool(provenance.get("is_synthetic", False))
    source = _html.escape(str(provenance.get("source", "")))
    warning = _html.escape(str(provenance.get("warning", "")))
    file_path = _html.escape(str(provenance.get("file_path", "") or "(unspecified)"))
    badge_text = "SYNTHETIC FIXTURE" if is_synth else "USER-PROVIDED CSV"
    badge_color = "#b7791f" if is_synth else "#2c5282"
    bg_color = "#fffbea" if is_synth else "#ebf8ff"
    border = "#f6ad55" if is_synth else "#90cdf4"
    return (
        '<div class="data-provenance" style="'
        'margin: 1.5rem 0; padding: 0.75rem 1rem;'
        ' background: {bg}; border: 2px solid {border};'
        ' border-radius: 6px; font-size: 0.9rem;'
        '">'
        '<div style="margin-bottom: 0.4rem;">'
        '<span style="display: inline-block; padding: 0.15rem 0.5rem;'
        ' background: {bc}; color: white;'
        ' border-radius: 4px; font-weight: 700;'
        ' font-size: 0.7rem; letter-spacing: 0.04em;">'
        'DATA: {bt}</span>'
        '<span style="margin-left: 0.6rem;">source = <code>{src}</code></span>'
        '</div>'
        '<div style="font-weight: 600; margin-bottom: 0.25rem;">{warn}</div>'
        '<div style="color: #555; font-size: 0.8rem;">'
        'file: <code>{fp}</code></div>'
        '</div>'
    ).format(bg=bg_color, border=border, bc=badge_color,
             bt=badge_text, src=source, warn=warning, fp=file_path)


def _inject_provenance_banner(html_str, provenance):
    """Splice the provenance banner into the rendered run-report HTML.

    Insertion point: immediately before the first `<h2>` header (i.e.
    between the `<h1>` + meta block and the Summary section). If no
    `<h2>` is found, the original HTML is returned unchanged.
    """
    banner = _provenance_banner_html(provenance)
    if not banner:
        return html_str
    marker = "<h2>"
    idx = html_str.find(marker)
    if idx == -1:
        return html_str
    return html_str[:idx] + banner + html_str[idx:]


def write_backtest_report(
    result, *,
    output_dir="reports",
    run_id=None,
    generated_at=None,
    notes=None,
    data_provenance=None,
):
    """Build then write JSON + static HTML reports.

    Both files are written atomically. The HTML is the standard
    `hermes.reports.render_html` output with the provenance banner
    spliced in just below the title.

    Returns:
      {"json_path": str, "html_path": str}
    """
    report = build_backtest_report(
        result,
        run_id=run_id,
        generated_at=generated_at,
        notes=notes,
        data_provenance=data_provenance,
    )

    abs_dir = os.path.abspath(output_dir)
    os.makedirs(abs_dir, exist_ok=True)
    json_path = os.path.join(abs_dir, REPORT_JSON_FILENAME)
    html_path = os.path.join(abs_dir, REPORT_HTML_FILENAME)

    # JSON: atomic via existing helper.
    write_json_atomic(json_path, report)

    # HTML: render via the unchanged `hermes.reports` renderer, splice
    # the provenance banner in, then write atomically.
    rendered = render_html(report)
    rendered = _inject_provenance_banner(rendered, report["data_provenance"])

    tmp = html_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(rendered)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, html_path)

    return {"json_path": json_path, "html_path": html_path}


__all__ = [
    "PHASE3A_REPORT_KEYS",
    "REPORT_HTML_FILENAME",
    "REPORT_JSON_FILENAME",
    "build_backtest_report",
    "write_backtest_report",
]
