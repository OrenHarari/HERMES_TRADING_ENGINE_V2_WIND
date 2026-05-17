"""Phase 3A - Offline backtest runner.

Smallest possible CSV-driven backtest harness that reuses
`safe_make_decision`, `CostModel`, and the existing performance report
without modifying any of them.

Design constraints (per Phase 3A spec):
  * Long-only.
  * One position at a time.
  * Sizing is fixed-fraction ONLY:
        notional   = equity * fixed_fraction
        shares     = notional / entry_price
    The core decision's `position_size` is recorded for audit/debug only
    in `core_position_size` and is NEVER treated as a dollar amount.
  * Entry at next bar's open (decision at bar i -> fill at bar i+1 open).
  * Exit checks per subsequent bar, in order:
        1. stop_loss   (low <= stop_price)        - WINS on within-bar tie
        2. take_profit (high >= take_profit_price)
        3. time_stop   (j - entry_index >= max_holding_bars) at close
        4. end_of_data on final bar at close if still open
  * Cost math is delegated to `apply_cost_model_to_trade` -- ZERO
    duplicated cost math here.
  * Deterministic: same (candles, config, signal_provider) -> same result.

The runner is intentionally minimal; risk_state, daily_loss tracking,
and consecutive-loss memory are NOT modeled in Phase 3A and are
explicitly out of scope. Each bar's `safe_make_decision` call uses a
fresh empty risk state, so risk-state-based blocks cannot fire.
"""

import math

from hermes.data.csv_loader import MODE_OHLCV_ONLY, MODE_WITH_SIGNALS
from hermes.integration import safe_make_decision
from hermes.safety.cost_model import CostModel, apply_cost_model_to_trade
from hermes.safety.data_contract import SYSTEM_MODE_BACKTEST
from hermes.signals.baseline_signal import baseline_signal

# ---- constants ----------------------------------------------------------

EXIT_STOP_LOSS = "stop_loss"
EXIT_TAKE_PROFIT = "take_profit"
EXIT_TIME_STOP = "time_stop"
EXIT_END_OF_DATA = "end_of_data"


# ---- config -------------------------------------------------------------

class BacktestConfig(object):
    """Frozen-ish config for the offline backtest runner.

    All attributes are validated in __init__ and then read-only in spirit.
    """

    __slots__ = (
        "symbol", "timeframe",
        "initial_equity",
        "fee_pct", "slippage_pct", "spread_pct",
        "take_profit_pct", "stop_loss_pct",
        "max_holding_bars",
        "fixed_fraction",
        "decision_config", "market_config",
    )

    def __init__(
        self,
        symbol="AMD",
        timeframe="1h",
        initial_equity=100_000.0,
        fee_pct=0.0005,
        slippage_pct=0.0005,
        spread_pct=0.0,
        take_profit_pct=0.02,
        stop_loss_pct=0.01,
        max_holding_bars=24,
        fixed_fraction=0.10,
        decision_config=None,
        market_config=None,
    ):
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(timeframe, str) or not timeframe:
            raise ValueError("timeframe must be a non-empty string")
        if not isinstance(initial_equity, (int, float)) or initial_equity <= 0:
            raise ValueError("initial_equity must be > 0")
        for name in ("fee_pct", "slippage_pct", "spread_pct"):
            v = locals()[name]
            if not isinstance(v, (int, float)) or v < 0 or v > 1:
                raise ValueError("{0} must be in [0, 1]".format(name))
        for name in ("take_profit_pct", "stop_loss_pct"):
            v = locals()[name]
            if not isinstance(v, (int, float)) or v <= 0 or v > 1:
                raise ValueError("{0} must be in (0, 1]".format(name))
        if not isinstance(max_holding_bars, int) or max_holding_bars < 1:
            raise ValueError("max_holding_bars must be a positive int")
        if (
            not isinstance(fixed_fraction, (int, float))
            or fixed_fraction <= 0
            or fixed_fraction > 1
        ):
            raise ValueError("fixed_fraction must be in (0, 1]")

        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_equity = float(initial_equity)
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.spread_pct = float(spread_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.max_holding_bars = int(max_holding_bars)
        self.fixed_fraction = float(fixed_fraction)
        self.decision_config = decision_config
        self.market_config = market_config

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "initial_equity": self.initial_equity,
            "fee_pct": self.fee_pct,
            "slippage_pct": self.slippage_pct,
            "spread_pct": self.spread_pct,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_holding_bars": self.max_holding_bars,
            "fixed_fraction": self.fixed_fraction,
        }


class OfflineBacktestResult(object):
    """Bundle of everything the runner produces."""

    __slots__ = (
        "config", "candles_count",
        "decisions", "completed_trades",
        "blocked_reasons_count", "equity_curve",
        "initial_equity", "final_equity",
        "net_pnl", "return_pct",
        "max_drawdown", "max_drawdown_pct",
        "start_timestamp", "end_timestamp",
        "cost_model_applied",
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))


# ---- signal provider helpers -------------------------------------------

def _signal_from_candle(candle):
    return {
        "sequence_value": candle["sequence_value"],
        "amd_value": candle["amd_value"],
        "combined_value": candle["combined_value"],
    }


def _resolve_signal_provider(candles, explicit_provider):
    """Pick the signal provider for this run.

    If `explicit_provider` is supplied, it is used as-is.
    Otherwise: if the first candle has signal columns, read them from the
    candle; else use baseline_signal.
    """
    if explicit_provider is not None:
        return explicit_provider, "explicit"
    if not candles:
        return baseline_signal, "baseline_signal"
    first = candles[0]
    if all(k in first for k in ("sequence_value", "amd_value", "combined_value")):
        return (lambda c, i: _signal_from_candle(c[i])), "csv_signals"
    return baseline_signal, "baseline_signal"


# ---- core loop ---------------------------------------------------------

def run_offline_backtest(candles, *, config=None, signal_provider=None):
    """Execute a deterministic offline backtest over `candles`.

    Parameters
    ----------
    candles : list of dict
        Validated candle dicts (output of `load_candles_csv`).
    config : BacktestConfig | None
        Runner config; defaults are AMD-1h sensible.
    signal_provider : callable | None
        Optional `(candles, current_index) -> dict` override. By default,
        embedded CSV signal columns are used if present; otherwise
        `baseline_signal` is injected.

    Returns
    -------
    OfflineBacktestResult
    """
    if not isinstance(candles, list) or len(candles) < 2:
        raise ValueError("candles must be a list with at least 2 entries")
    if config is None:
        config = BacktestConfig()
    if not isinstance(config, BacktestConfig):
        raise ValueError("config must be a BacktestConfig instance")

    sig_fn, _sig_kind = _resolve_signal_provider(candles, signal_provider)

    cost_model = CostModel(
        fee_pct=config.fee_pct,
        slippage_pct=config.slippage_pct,
        spread_pct=config.spread_pct,
    )

    decisions = []
    completed_trades = []
    blocked_reasons = {}
    equity_curve = [{
        "timestamp": int(candles[0]["timestamp"]),
        "equity": float(config.initial_equity),
    }]
    equity = float(config.initial_equity)

    open_position = None  # dict when in-flight, else None

    n = len(candles)
    last_index = n - 1

    for i in range(n):
        candle = candles[i]

        # ---- (A) handle open position FIRST: check exit on this bar ---
        if open_position is not None and i > open_position["entry_index"]:
            exit_ts = int(candle["timestamp"])
            entry_index = open_position["entry_index"]
            entry_price = open_position["entry_price"]
            shares = open_position["shares"]
            stop_price = open_position["stop_price"]
            tp_price = open_position["take_profit_price"]

            held_bars = i - entry_index
            stop_hit = candle["low"] <= stop_price
            tp_hit = candle["high"] >= tp_price
            exit_reason = None
            exit_price = None
            if stop_hit:
                # Conservative tie-break: stop wins.
                exit_reason = EXIT_STOP_LOSS
                exit_price = stop_price
            elif tp_hit:
                exit_reason = EXIT_TAKE_PROFIT
                exit_price = tp_price
            elif held_bars >= config.max_holding_bars:
                exit_reason = EXIT_TIME_STOP
                exit_price = float(candle["close"])
            elif i == last_index:
                exit_reason = EXIT_END_OF_DATA
                exit_price = float(candle["close"])

            if exit_reason is not None:
                trade = _close_position(
                    open_position, exit_price, exit_ts, exit_reason,
                    cost_model,
                )
                completed_trades.append(trade)
                equity += trade["net_pnl"]
                equity_curve.append({
                    "timestamp": exit_ts,
                    "equity": equity,
                })
                open_position = None

        # ---- (B) make a decision (only used for new entries) -----------
        if i == last_index:
            # Cannot open at the very last bar (no next bar to fill).
            continue

        signal = sig_fn(candles, i)
        decision = safe_make_decision(
            signal,
            candles,
            i,
            system_mode=SYSTEM_MODE_BACKTEST,
            now_ts=int(candle["timestamp"]),
            decision_config=config.decision_config,
            market_config=config.market_config,
        )
        decisions.append(decision)

        if not decision["trade_allowed"]:
            reason = decision.get("reason", "") or "unknown"
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
            continue

        # Already in a position -> cannot open another.
        if open_position is not None:
            blocked_reasons["already_in_position"] = (
                blocked_reasons.get("already_in_position", 0) + 1
            )
            continue

        # ---- (C) open a new position at next bar's open ---------------
        next_candle = candles[i + 1]
        entry_price = float(next_candle["open"])
        if entry_price <= 0:
            continue
        notional = equity * config.fixed_fraction
        shares = notional / entry_price
        if shares <= 0:
            continue

        stop_price = entry_price * (1.0 - config.stop_loss_pct)
        tp_price = entry_price * (1.0 + config.take_profit_pct)

        open_position = {
            "entry_index": i + 1,
            "entry_timestamp": int(next_candle["timestamp"]),
            "decision_index": i,
            "entry_price": entry_price,
            "shares": shares,
            "notional": notional,
            "stop_price": stop_price,
            "take_profit_price": tp_price,
            "regime": decision.get("regime", ""),
            "confidence": decision.get("confidence", 0.0),
            "agreement": decision.get("agreement", 0.0),
            "core_position_size": decision.get("position_size", 0.0),
            "signal": dict(signal),
        }

    # Final equity & metrics.
    final_equity = equity
    net_pnl = final_equity - config.initial_equity
    return_pct = (
        (net_pnl / config.initial_equity) * 100.0
        if config.initial_equity > 0 else 0.0
    )
    max_dd, max_dd_pct = _compute_max_drawdown(equity_curve)

    return OfflineBacktestResult(
        config=config,
        candles_count=n,
        decisions=decisions,
        completed_trades=completed_trades,
        blocked_reasons_count=blocked_reasons,
        equity_curve=equity_curve,
        initial_equity=float(config.initial_equity),
        final_equity=float(final_equity),
        net_pnl=float(net_pnl),
        return_pct=float(return_pct),
        max_drawdown=float(max_dd),
        max_drawdown_pct=float(max_dd_pct),
        start_timestamp=int(candles[0]["timestamp"]),
        end_timestamp=int(candles[-1]["timestamp"]),
        cost_model_applied=True,
    )


# ---- helpers -----------------------------------------------------------

def _close_position(open_position, exit_price, exit_ts, exit_reason, cost_model):
    """Build the completed-trade record using the existing CostModel API.

    NO duplicated cost math here: `apply_cost_model_to_trade` is the
    single authority for fees/slippage/net_pnl computation.
    """
    base = {
        "entry_price": float(open_position["entry_price"]),
        "exit_price": float(exit_price),
        "position_size": float(open_position["shares"]),  # CostModel.apply()
                                                          # uses position_size.
        "entry_timestamp": int(open_position["entry_timestamp"]),
        "exit_timestamp": int(exit_ts),
        "regime": str(open_position["regime"]),
        "confidence": float(open_position["confidence"]),
        "agreement": float(open_position["agreement"]),
        "exit_reason": str(exit_reason),
        "shares": float(open_position["shares"]),
        "notional": float(open_position["notional"]),
        "core_position_size": float(open_position["core_position_size"]),
        "sequence_value": float(open_position["signal"]["sequence_value"]),
        "amd_value": float(open_position["signal"]["amd_value"]),
        "combined_value": float(open_position["signal"]["combined_value"]),
    }
    enriched = apply_cost_model_to_trade(base, cost_model)
    enriched["pnl"] = enriched["net_pnl"]  # alias for older consumers
    enriched["outcome"] = "win" if enriched["net_pnl"] > 0 else (
        "loss" if enriched["net_pnl"] < 0 else "neutral"
    )
    return enriched


def _compute_max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0, 0.0
    peak = equity_curve[0]["equity"]
    max_dd = 0.0
    max_dd_pct = 0.0
    for point in equity_curve:
        eq = point["equity"]
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = (dd / peak) * 100.0 if peak > 0 else 0.0
    return max_dd, max_dd_pct


# Re-exports for testing / dependency-injection from the data layer.
__all__ = [
    "EXIT_STOP_LOSS",
    "EXIT_TAKE_PROFIT",
    "EXIT_TIME_STOP",
    "EXIT_END_OF_DATA",
    "BacktestConfig",
    "OfflineBacktestResult",
    "run_offline_backtest",
    "MODE_OHLCV_ONLY",
    "MODE_WITH_SIGNALS",
]
