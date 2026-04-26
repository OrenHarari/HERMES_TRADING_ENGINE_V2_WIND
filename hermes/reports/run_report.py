"""Run-report builder + writer.

Three pure functions:

  build_run_report(...) -> dict          # canonical run-report shape
  render_html(report)   -> str           # self-contained static HTML page
  write_run_report(report, output_dir)   # writes JSON + HTML atomically

The report aggregates whatever the caller has on hand: decisions (e.g.
output of `safe_make_decision`), completed trade records, a performance
report, the learning summary, and kill-switch state. Every section is
optional; missing sections render as "(none)".

No web server, no template engine, no JavaScript, no external dependencies.
"""

import html
import json
import os
from datetime import datetime, timezone

from hermes.utils.json_io import write_json_atomic

# ---- canonical layout ---------------------------------------------------
DEFAULT_OUTPUT_DIR = "reports"
REPORT_JSON_FILENAME = "latest_run.json"
REPORT_HTML_FILENAME = "latest_run.html"

REQUIRED_REPORT_KEYS = (
    "run_id",
    "generated_at",
    "system_mode",
    "decisions",
    "completed_trades",
    "performance_report",
    "learning_summary",
    "kill_switch_state",
    "summary",
    "notes",
)


def _utc_iso_now():
    return datetime.now(timezone.utc).isoformat()


def _coerce_list(v):
    return list(v) if v is not None else []


def _summary(decisions, completed_trades):
    approved = 0
    blocked = 0
    for d in decisions:
        if isinstance(d, dict) and d.get("trade_allowed") is True:
            approved += 1
        else:
            blocked += 1
    return {
        "total_decisions": len(decisions),
        "approved_decisions": approved,
        "blocked_decisions": blocked,
        "total_trades": len(completed_trades),
    }


def build_run_report(
    *,
    decisions=None,
    completed_trades=None,
    performance_report=None,
    learning_summary=None,
    kill_switch_state=None,
    system_mode=None,
    run_id=None,
    notes=None,
    generated_at=None,
):
    """Build the canonical run-report dict.

    All inputs are optional. If `generated_at` is None, the current UTC
    timestamp is used; pass an explicit string for deterministic tests.
    `run_id` defaults to the generated_at timestamp.
    """
    decisions_list = _coerce_list(decisions)
    trades_list = _coerce_list(completed_trades)
    gen_at = generated_at if generated_at is not None else _utc_iso_now()
    report = {
        "run_id": str(run_id) if run_id is not None else gen_at,
        "generated_at": gen_at,
        "system_mode": str(system_mode) if system_mode is not None else "",
        "decisions": decisions_list,
        "completed_trades": trades_list,
        "performance_report": (
            dict(performance_report) if isinstance(performance_report, dict) else {}
        ),
        "learning_summary": (
            dict(learning_summary) if isinstance(learning_summary, dict) else {}
        ),
        "kill_switch_state": (
            dict(kill_switch_state) if isinstance(kill_switch_state, dict) else {}
        ),
        "summary": _summary(decisions_list, trades_list),
        "notes": str(notes) if notes is not None else "",
    }
    return report


# ---- HTML rendering -----------------------------------------------------
_HTML_HEAD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HERMES Run Report &mdash; {run_id}</title>
<style>
  :root {{
    --fg: #1a1a1a; --bg: #fdfdfd; --muted: #555; --accent: #2c5282;
    --good: #2f855a; --bad: #c53030; --warn: #b7791f; --border: #e2e8f0;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         Helvetica, Arial, sans-serif; background: var(--bg); color: var(--fg);
         margin: 2rem auto; max-width: 1200px; padding: 0 1.5rem;
         line-height: 1.4; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 0.25rem; }}
  .meta {{ color: var(--muted); font-size: 0.875rem; margin-bottom: 2rem; }}
  h2 {{ font-size: 1.125rem; margin: 2rem 0 0.5rem;
       padding-bottom: 0.25rem; border-bottom: 1px solid var(--border); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem;
          margin-top: 0.5rem; }}
  th, td {{ padding: 0.4rem 0.65rem; text-align: left;
           border-bottom: 1px solid var(--border); vertical-align: top; }}
  th {{ background: #f7fafc; font-weight: 600; color: var(--muted);
       font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  tr:hover td {{ background: #f7fafc; }}
  .pass {{ color: var(--good); font-weight: 600; }}
  .fail {{ color: var(--bad); font-weight: 600; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .empty {{ color: var(--muted); font-style: italic; padding: 0.5rem 0; }}
  .kv {{ display: grid; grid-template-columns: max-content 1fr;
        gap: 0.25rem 1rem; font-size: 0.875rem;
        font-variant-numeric: tabular-nums; }}
  .kv .k {{ color: var(--muted); }}
  .badge {{ display: inline-block; padding: 0.125rem 0.5rem; border-radius: 4px;
           font-size: 0.7rem; font-weight: 600; background: var(--accent);
           color: white; text-transform: uppercase; letter-spacing: 0.04em; }}
  .badge.warn {{ background: var(--warn); }}
  .badge.bad {{ background: var(--bad); }}
  pre {{ background: #f7fafc; padding: 0.75rem; border-radius: 4px;
        font-size: 0.75rem; overflow-x: auto; margin: 0.5rem 0;
        border: 1px solid var(--border); white-space: pre-wrap; }}
  .footer {{ margin-top: 3rem; color: var(--muted); font-size: 0.75rem;
            text-align: center; }}
</style>
</head>
<body>
<h1>HERMES Run Report</h1>
<div class="meta">
  <strong>{run_id}</strong>
  &middot; generated: {generated_at}
  &middot; mode: <span class="badge">{system_mode}</span>
</div>
"""

_HTML_FOOT = (
    '<div class="footer">HERMES Trading Engine v2 &mdash; '
    'static report, no JavaScript, no external dependencies.</div>'
    '\n</body>\n</html>\n'
)


def _esc(v):
    """HTML-escape any value, coerced via str()."""
    return html.escape(str(v), quote=True)


def _format_num(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        if v == float("inf"):
            return "&infin;"
        if v == float("-inf"):
            return "-&infin;"
        return "{:.6g}".format(v)
    return _esc(v)


def _section_summary(report):
    s = report.get("summary", {})
    return (
        '<div class="kv">'
        '<div class="k">Total decisions</div><div class="v">{td}</div>'
        '<div class="k">Approved</div><div class="v pass">{ad}</div>'
        '<div class="k">Blocked</div><div class="v fail">{bd}</div>'
        '<div class="k">Completed trades</div><div class="v">{tt}</div>'
        '</div>'
    ).format(
        td=_format_num(s.get("total_decisions", 0)),
        ad=_format_num(s.get("approved_decisions", 0)),
        bd=_format_num(s.get("blocked_decisions", 0)),
        tt=_format_num(s.get("total_trades", 0)),
    )


def _section_performance(report):
    perf = report.get("performance_report", {})
    if not perf:
        return '<div class="empty">(no performance report)</div>'
    rows = []
    for k in sorted(perf.keys()):
        v = perf[k]
        if isinstance(v, dict):
            v_html = "<pre>{}</pre>".format(
                _esc(json.dumps(v, sort_keys=True, indent=2))
            )
        else:
            v_html = '<span class="num">{}</span>'.format(_format_num(v))
        rows.append(
            '<tr><td>{k}</td><td>{v}</td></tr>'.format(k=_esc(k), v=v_html)
        )
    return (
        '<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>'
        '<tbody>{}</tbody></table>'.format("".join(rows))
    )


def _decision_row(d):
    if not isinstance(d, dict):
        return '<tr><td colspan="6"><pre>{}</pre></td></tr>'.format(_esc(d))
    cls = "pass" if d.get("trade_allowed") else "fail"
    label = "PASS" if d.get("trade_allowed") else "BLOCKED"
    return (
        '<tr>'
        '<td><span class="{cls}">{label}</span></td>'
        '<td class="num">{conf}</td>'
        '<td class="num">{agree}</td>'
        '<td>{regime}</td>'
        '<td class="num">{size}</td>'
        '<td>{reason}</td>'
        '</tr>'
    ).format(
        cls=cls,
        label=label,
        conf=_format_num(d.get("confidence", 0.0)),
        agree=_format_num(d.get("agreement", 0.0)),
        regime=_esc(d.get("regime", "")),
        size=_format_num(d.get("position_size", 0.0)),
        reason=_esc(d.get("reason", "")),
    )


def _section_decisions(report):
    decisions = report.get("decisions", [])
    if not decisions:
        return '<div class="empty">(no decisions)</div>'
    rows = "".join(_decision_row(d) for d in decisions)
    return (
        '<table><thead><tr>'
        '<th>Status</th><th>Confidence</th><th>Agreement</th>'
        '<th>Regime</th><th>Size</th><th>Reason</th>'
        '</tr></thead><tbody>{}</tbody></table>'.format(rows)
    )


def _trade_row(t):
    if not isinstance(t, dict):
        return '<tr><td colspan="6"><pre>{}</pre></td></tr>'.format(_esc(t))
    outcome = t.get("outcome", "")
    cls = {"win": "pass", "loss": "fail"}.get(outcome, "")
    return (
        '<tr>'
        '<td><span class="{cls}">{outcome}</span></td>'
        '<td>{regime}</td>'
        '<td class="num">{entry}</td>'
        '<td class="num">{exit_}</td>'
        '<td class="num">{net}</td>'
        '<td>{reason}</td>'
        '</tr>'
    ).format(
        cls=cls,
        outcome=_esc(outcome),
        regime=_esc(t.get("regime", "")),
        entry=_format_num(t.get("entry_price", 0.0)),
        exit_=_format_num(t.get("exit_price", 0.0)),
        net=_format_num(t.get("net_pnl", t.get("pnl", 0.0))),
        reason=_esc(t.get("exit_reason", "")),
    )


def _section_trades(report):
    trades = report.get("completed_trades", [])
    if not trades:
        return '<div class="empty">(no completed trades)</div>'
    rows = "".join(_trade_row(t) for t in trades)
    return (
        '<table><thead><tr>'
        '<th>Outcome</th><th>Regime</th><th>Entry</th>'
        '<th>Exit</th><th>Net&nbsp;PnL</th><th>Exit&nbsp;Reason</th>'
        '</tr></thead><tbody>{}</tbody></table>'.format(rows)
    )


def _section_learning(report):
    ls = report.get("learning_summary", {})
    if not ls:
        return '<div class="empty">(no learning summary)</div>'
    flat_rows = []
    for k in sorted(ls.keys()):
        v = ls[k]
        if isinstance(v, (dict, list)):
            v_html = "<pre>{}</pre>".format(
                _esc(json.dumps(v, sort_keys=True, indent=2, default=str))
            )
        else:
            v_html = '<span class="num">{}</span>'.format(_format_num(v))
        flat_rows.append(
            '<tr><td>{k}</td><td>{v}</td></tr>'.format(k=_esc(k), v=v_html)
        )
    return (
        '<table><thead><tr><th>Field</th><th>Value</th></tr></thead>'
        '<tbody>{}</tbody></table>'.format("".join(flat_rows))
    )


def _section_kill_switch(report):
    ks = report.get("kill_switch_state", {})
    if not ks:
        return '<div class="empty">(no kill switch state recorded)</div>'
    active = bool(ks.get("active", False))
    badge = '<span class="badge bad">ACTIVE</span>' if active else (
        '<span class="badge">INACTIVE</span>'
    )
    reason = _esc(ks.get("reason", "")) or "(none)"
    activated_ts = _esc(ks.get("activated_ts", ""))
    extra = ""
    if "log" in ks and isinstance(ks["log"], list) and ks["log"]:
        extra = "<pre>{}</pre>".format(
            _esc(json.dumps(ks["log"], sort_keys=True, indent=2, default=str))
        )
    return (
        '<div class="kv">'
        '<div class="k">Status</div><div class="v">{badge}</div>'
        '<div class="k">Reason</div><div class="v">{reason}</div>'
        '<div class="k">Activated</div><div class="v">{ts}</div>'
        '</div>{extra}'
    ).format(badge=badge, reason=reason, ts=activated_ts, extra=extra)


def _section_notes(report):
    notes = report.get("notes", "")
    if not notes:
        return ""
    return (
        '<h2>Notes</h2><pre>{}</pre>'.format(_esc(notes))
    )


def render_html(report):
    """Render the report dict as a single self-contained HTML page."""
    if not isinstance(report, dict):
        raise ValueError("report must be a dict")
    head = _HTML_HEAD_TEMPLATE.format(
        run_id=_esc(report.get("run_id", "")),
        generated_at=_esc(report.get("generated_at", "")),
        system_mode=_esc(report.get("system_mode", "") or "(none)"),
    )
    body_parts = [
        head,
        "<h2>Summary</h2>", _section_summary(report),
        "<h2>Performance</h2>", _section_performance(report),
        "<h2>Decisions</h2>", _section_decisions(report),
        "<h2>Completed Trades</h2>", _section_trades(report),
        "<h2>Learning Summary</h2>", _section_learning(report),
        "<h2>Kill Switch</h2>", _section_kill_switch(report),
        _section_notes(report),
        _HTML_FOOT,
    ]
    return "".join(body_parts)


def write_run_report(report, output_dir=DEFAULT_OUTPUT_DIR):
    """Write `latest_run.json` + `latest_run.html` to `output_dir`.

    Both files are written atomically (write-temp-then-rename for JSON;
    direct write-then-rename for HTML). Returns:
      {"json_path": str, "html_path": str}
    """
    if not isinstance(report, dict):
        raise ValueError("report must be a dict")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")
    abs_dir = os.path.abspath(output_dir)
    if not os.path.isdir(abs_dir):
        os.makedirs(abs_dir, exist_ok=True)

    json_path = os.path.join(abs_dir, REPORT_JSON_FILENAME)
    html_path = os.path.join(abs_dir, REPORT_HTML_FILENAME)

    write_json_atomic(json_path, report)

    # HTML: write to .tmp then rename, parallel to write_json_atomic.
    tmp = html_path + ".tmp"
    rendered = render_html(report)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(rendered)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, html_path)

    return {"json_path": json_path, "html_path": html_path}


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "REPORT_HTML_FILENAME",
    "REPORT_JSON_FILENAME",
    "REQUIRED_REPORT_KEYS",
    "build_run_report",
    "render_html",
    "write_run_report",
]
