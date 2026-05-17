"""Phase 3A - offline backtest harness.

Stand-alone runner that consumes a list of validated candles, calls the
existing `safe_make_decision` per bar, and produces a deterministic
result with completed trades + equity curve. No external dependencies.
"""

from hermes.backtest.offline_runner import (
    EXIT_END_OF_DATA,
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT,
    EXIT_TIME_STOP,
    BacktestConfig,
    OfflineBacktestResult,
    run_offline_backtest,
)

__all__ = [
    "EXIT_END_OF_DATA",
    "EXIT_STOP_LOSS",
    "EXIT_TAKE_PROFIT",
    "EXIT_TIME_STOP",
    "BacktestConfig",
    "OfflineBacktestResult",
    "run_offline_backtest",
]
