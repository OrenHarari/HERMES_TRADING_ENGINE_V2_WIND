"""Phase 3A - CLI for the offline backtest runner.

Pure stdlib (argparse + json). No live, no broker, no network.

AMD is the engineering test target only because it is liquid and
volatile enough to exercise the pipeline. This is NOT an investment
recommendation.

Usage:
    python run_offline_backtest.py --csv path/to/AMD_1h.csv \\
        --symbol AMD --timeframe 1h \\
        --initial-equity 100000 \\
        --fee-pct 0.0005 --slippage-pct 0.0005 \\
        --take-profit 0.02 --stop-loss 0.01 \\
        --output-dir reports

Outputs:
    reports/latest_run.json
    reports/latest_run.html
"""

import argparse
import sys

from hermes.backtest.offline_runner import BacktestConfig, run_offline_backtest
from hermes.backtest.report import write_backtest_report
from hermes.data.csv_loader import load_candles_csv
from hermes.data.provenance import (
    DATA_SOURCE_TEST_FIXTURE,
    DATA_SOURCE_USER_PROVIDED,
    VALID_SOURCES,
    build_data_provenance,
)


def _build_parser():
    p = argparse.ArgumentParser(
        prog="run_offline_backtest",
        description=(
            "HERMES V2 - offline CSV backtest. Reads a local OHLCV (or "
            "OHLCV+signals) CSV, runs the existing safe_make_decision "
            "pipeline bar-by-bar, and writes a JSON + static HTML report. "
            "No network. No broker. No live execution."
        ),
    )
    p.add_argument("--csv", required=True, help="Path to candle CSV file.")
    p.add_argument("--symbol", default="AMD",
                   help="Symbol label for the report (default: AMD).")
    p.add_argument("--timeframe", default="1h",
                   help="Timeframe label for the report (default: 1h).")
    p.add_argument("--initial-equity", type=float, default=100_000.0,
                   dest="initial_equity",
                   help="Starting equity (default: 100000).")
    p.add_argument("--fee-pct", type=float, default=0.0005, dest="fee_pct",
                   help="Per-side fee fraction (default: 0.0005 = 5 bps).")
    p.add_argument("--slippage-pct", type=float, default=0.0005,
                   dest="slippage_pct",
                   help="Per-side slippage fraction (default: 0.0005).")
    p.add_argument("--spread-pct", type=float, default=0.0,
                   dest="spread_pct",
                   help="Spread fraction (default: 0.0).")
    p.add_argument("--take-profit", type=float, default=0.02,
                   dest="take_profit_pct",
                   help="Take-profit fraction (default: 0.02 = 2%%).")
    p.add_argument("--stop-loss", type=float, default=0.01,
                   dest="stop_loss_pct",
                   help="Stop-loss fraction (default: 0.01 = 1%%).")
    p.add_argument("--max-holding-bars", type=int, default=24,
                   dest="max_holding_bars",
                   help="Time-stop in bars (default: 24).")
    p.add_argument("--fixed-fraction", type=float, default=0.10,
                   dest="fixed_fraction",
                   help="Fraction of equity per trade (default: 0.10).")
    p.add_argument("--output-dir", default="reports", dest="output_dir",
                   help="Directory for the JSON+HTML report (default: reports).")
    p.add_argument("--run-id", default=None, dest="run_id",
                   help="Optional report run_id; defaults to a timestamp.")
    p.add_argument(
        "--source", default=DATA_SOURCE_USER_PROVIDED,
        choices=list(VALID_SOURCES), dest="source",
        help=(
            "Data provenance for the input CSV. Default: '"
            + DATA_SOURCE_USER_PROVIDED + "'. Use '"
            + DATA_SOURCE_TEST_FIXTURE
            + "' ONLY for hand-crafted CSV fixtures under "
            "tests/fixtures/. NEVER claim downloaded or scraped data is "
            "synthetic, and NEVER claim a synthetic fixture is real."
        ),
    )
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    loaded = load_candles_csv(args.csv)
    config = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_equity=args.initial_equity,
        fee_pct=args.fee_pct,
        slippage_pct=args.slippage_pct,
        spread_pct=args.spread_pct,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        max_holding_bars=args.max_holding_bars,
        fixed_fraction=args.fixed_fraction,
    )
    result = run_offline_backtest(loaded["candles"], config=config)
    provenance = build_data_provenance(
        file_path=args.csv,
        symbol=config.symbol,
        timeframe=config.timeframe,
        row_count=loaded["row_count"],
        start_timestamp=int(loaded["first_timestamp"]),
        end_timestamp=int(loaded["last_timestamp"]),
        source=args.source,
    )
    paths = write_backtest_report(
        result,
        output_dir=args.output_dir,
        run_id=args.run_id,
        data_provenance=provenance,
        notes=(
            "AMD is selected as an engineering test target only. This is "
            "NOT an investment recommendation."
        ),
    )

    sys.stdout.write(
        "\n=== HERMES V2 - Offline Backtest ===\n"
        "Symbol            : {symbol}\n"
        "Timeframe         : {tf}\n"
        "Candles loaded    : {n}\n"
        "CSV mode          : {mode}\n"
        "Initial equity    : ${ie:.2f}\n"
        "Final equity      : ${fe:.2f}\n"
        "Net PnL           : ${pnl:.2f}\n"
        "Return            : {ret:.4f}%\n"
        "Trades completed  : {tc}\n"
        "Max drawdown      : ${dd:.2f} ({dd_pct:.4f}%)\n"
        "Cost model        : applied\n"
        "Data source       : {ds}\n"
        "Provenance        : {pw}\n"
        "JSON report       : {jp}\n"
        "HTML report       : {hp}\n".format(
            symbol=config.symbol,
            tf=config.timeframe,
            n=loaded["row_count"],
            mode=loaded["mode"],
            ie=result.initial_equity,
            fe=result.final_equity,
            pnl=result.net_pnl,
            ret=result.return_pct,
            tc=len(result.completed_trades),
            dd=result.max_drawdown,
            dd_pct=result.max_drawdown_pct,
            ds=provenance["source"],
            pw=provenance["warning"],
            jp=paths["json_path"],
            hp=paths["html_path"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
