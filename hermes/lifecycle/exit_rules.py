"""Prompt 2 / Step 5B - Trade lifecycle: deterministic exit-decision rules.

Pure, deterministic exit-decision function. Inspects ONLY the open position,
the present candle, the present index, the present combined_value, and two
external flags (risk_blocked, end_of_backtest). Never reads future candles.

Priority order (deterministic):
  1. risk_guardrail
  2. end_of_backtest
  3. stop_loss
  4. take_profit
  5. max_holding_bars
  6. signal_decay
"""

from hermes.utils.bounds import is_real_number, is_unit_interval

EXIT_REASON_STOP_LOSS = "stop_loss"
EXIT_REASON_TAKE_PROFIT = "take_profit"
EXIT_REASON_MAX_HOLDING = "max_holding_bars"
EXIT_REASON_SIGNAL_DECAY = "signal_decay"
EXIT_REASON_RISK_GUARDRAIL = "risk_guardrail"
EXIT_REASON_END_OF_BACKTEST = "end_of_backtest"

EXIT_REASONS = (
    EXIT_REASON_STOP_LOSS,
    EXIT_REASON_TAKE_PROFIT,
    EXIT_REASON_MAX_HOLDING,
    EXIT_REASON_SIGNAL_DECAY,
    EXIT_REASON_RISK_GUARDRAIL,
    EXIT_REASON_END_OF_BACKTEST,
)


class ExitRulesConfig(object):
    """Configurable thresholds for the exit-decision rules.

    Attributes:
      stop_distance:           absolute price distance for stop_loss
                               (entry_price - distance).
      take_profit_distance:    absolute price distance for take_profit.
      max_holding_bars:        bars-held threshold for max_holding rule.
      signal_decay_threshold:  combined_value floor; below -> exit. Spec
                               default is 0.30.
    """

    __slots__ = (
        "stop_distance",
        "take_profit_distance",
        "max_holding_bars",
        "signal_decay_threshold",
    )

    def __init__(
        self,
        stop_distance=None,
        take_profit_distance=None,
        max_holding_bars=None,
        signal_decay_threshold=0.30,
    ):
        if stop_distance is not None:
            if not is_real_number(stop_distance) or stop_distance < 0.0:
                raise ValueError("stop_distance must be a non-negative number")
        if take_profit_distance is not None:
            if (
                not is_real_number(take_profit_distance)
                or take_profit_distance < 0.0
            ):
                raise ValueError(
                    "take_profit_distance must be a non-negative number"
                )
        if max_holding_bars is not None:
            if (
                not isinstance(max_holding_bars, int)
                or isinstance(max_holding_bars, bool)
                or max_holding_bars < 0
            ):
                raise ValueError("max_holding_bars must be a non-negative int")
        if not is_unit_interval(signal_decay_threshold):
            raise ValueError("signal_decay_threshold must be in [0, 1]")
        self.stop_distance = (
            float(stop_distance) if stop_distance is not None else None
        )
        self.take_profit_distance = (
            float(take_profit_distance)
            if take_profit_distance is not None
            else None
        )
        self.max_holding_bars = max_holding_bars
        self.signal_decay_threshold = float(signal_decay_threshold)


def _validate_inputs(position, current_candle, current_index):
    # Avoid a circular import; lazy import within function.
    from hermes.lifecycle.lifecycle import OpenPosition

    if not isinstance(position, OpenPosition):
        raise ValueError("position must be an OpenPosition instance")
    if not isinstance(current_candle, dict):
        raise ValueError("current_candle must be a dict")
    if "close" not in current_candle:
        raise ValueError("current_candle missing 'close'")
    if not is_real_number(current_candle["close"]):
        raise ValueError("current_candle['close'] must be numeric")
    if (
        not isinstance(current_index, int)
        or isinstance(current_index, bool)
        or current_index < 0
    ):
        raise ValueError("current_index must be a non-negative int")
    if current_index < position.entry_index:
        raise ValueError("current_index must be >= position.entry_index")


def decide_exit(
    position,
    current_candle,
    current_index,
    current_combined_value=None,
    risk_blocked=False,
    end_of_backtest=False,
    config=None,
):
    """Pure exit-decision function.

    Returns one of:
      {"should_exit": False, "bars_held": int}
      {"should_exit": True, "exit_reason": str, "exit_price": float,
       "bars_held": int}
    """
    _validate_inputs(position, current_candle, current_index)
    if config is None:
        config = ExitRulesConfig()
    if not isinstance(risk_blocked, bool):
        raise ValueError("risk_blocked must be bool")
    if not isinstance(end_of_backtest, bool):
        raise ValueError("end_of_backtest must be bool")
    if current_combined_value is not None and not is_unit_interval(
        current_combined_value
    ):
        raise ValueError("current_combined_value must be in [0, 1] or None")

    bars_held = int(current_index - position.entry_index)
    exit_price = float(current_candle["close"])

    if risk_blocked:
        return _exit(EXIT_REASON_RISK_GUARDRAIL, exit_price, bars_held)
    if end_of_backtest:
        return _exit(EXIT_REASON_END_OF_BACKTEST, exit_price, bars_held)

    if (
        config.stop_distance is not None
        and exit_price <= position.entry_price - config.stop_distance
    ):
        return _exit(EXIT_REASON_STOP_LOSS, exit_price, bars_held)

    if (
        config.take_profit_distance is not None
        and exit_price >= position.entry_price + config.take_profit_distance
    ):
        return _exit(EXIT_REASON_TAKE_PROFIT, exit_price, bars_held)

    if (
        config.max_holding_bars is not None
        and bars_held >= config.max_holding_bars
    ):
        return _exit(EXIT_REASON_MAX_HOLDING, exit_price, bars_held)

    if (
        current_combined_value is not None
        and current_combined_value < config.signal_decay_threshold
    ):
        return _exit(EXIT_REASON_SIGNAL_DECAY, exit_price, bars_held)

    return {"should_exit": False, "bars_held": bars_held}


def _exit(reason, exit_price, bars_held):
    return {
        "should_exit": True,
        "exit_reason": reason,
        "exit_price": float(exit_price),
        "bars_held": int(bars_held),
    }
