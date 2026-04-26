"""Run reporting layer.

Pure-stdlib report generation: build a canonical run-report dict and emit
`reports/latest_run.json` + `reports/latest_run.html`. No web server, no
templates, no JS framework, no external dependency.
"""

from hermes.reports.run_report import (
    DEFAULT_OUTPUT_DIR,
    REPORT_HTML_FILENAME,
    REPORT_JSON_FILENAME,
    REQUIRED_REPORT_KEYS,
    build_run_report,
    render_html,
    write_run_report,
)

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "REPORT_HTML_FILENAME",
    "REPORT_JSON_FILENAME",
    "REQUIRED_REPORT_KEYS",
    "build_run_report",
    "render_html",
    "write_run_report",
]
