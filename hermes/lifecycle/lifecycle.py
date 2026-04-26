"""Prompt 2 / Step 5B - Trade lifecycle FSM.

Adds an explicit OpenPosition state object plus a `complete_trade` helper
that produces the canonical Phase-1 completed-trade record EXTENDED with the
Phase-2 exit fields:

  exit_reason, exit_price, bars_held, pnl, fees, slippage, spread_cost,
  net_pnl

The Phase-1 minimal builder (`hermes.lifecycle.completed_trade`) is reused
unchanged; this module augments its output additively.

Lifecycle:
  signal -> entry decision -> risk validation -> OpenPosition
        -> decide_exit (deterministic) -> complete_trade -> memory log

No completed trade may exist without entry timestamp, exit timestamp, entry
price, exit price, exit reason, and net_pnl. `complete_trade` raises if
required information is missing.
"""

from hermes.lifecycle.completed_trade import (
    REQUIRED_TRADE_KEYS,
    build_completed_trade,
    is_complete_trade_record,
)
from hermes.utils.bounds import is_real_number

REQUIRED_OPEN_POSITION_META_KEYS = (
    "date",
    "hour",
    "sequence_value",
    "amd_value",
    "combined_value",
    "agreement",
    "confidence",
    "regime",
    "momentum_score",
    "volatility_score",
)

REQUIRED_EXIT_FIELDS = (
    "exit_reason",
    "exit_price",
    "bars_held",
    "pnl",
    "fees",
    "slippage",
    "spread_cost",
    "net_pnl",
)


class OpenPosition(object):
    """Mutable, explicit state for a position between entry and exit.

    Note: this class never mutates after construction in Step 5B; it is a
    container for the entry-time information required to later build the
    completed-trade record. Future Prompt 2 steps may add accessors but must
    not change existing fields.
    """

    __slots__ = (
        "entry_timestamp",
        "entry_index",
        "entry_price",
        "position_size",
        "entry_signal_meta",
    )

    def __init__(
        self,
        entry_timestamp,
        entry_index,
        entry_price,
        position_size,
        entry_signal_meta,
    ):
        if not is_real_number(entry_timestamp) and not isinstance(
            entry_timestamp, str
        ):
            raise ValueError("entry_timestamp must be numeric or str")
        if (
            not isinstance(entry_index, int)
            or isinstance(entry_index, bool)
            or entry_index < 0
        ):
            raise ValueError("entry_index must be a non-negative int")
        if not is_real_number(entry_price) or entry_price <= 0.0:
            raise ValueError("entry_price must be > 0")
        if not is_real_number(position_size) or position_size < 0.0:
            raise ValueError("position_size must be >= 0")
        if not isinstance(entry_signal_meta, dict):
            raise ValueError("entry_signal_meta must be a dict")
        for k in REQUIRED_OPEN_POSITION_META_KEYS:
            if k not in entry_signal_meta:
                raise ValueError(
                    "entry_signal_meta missing required key: {!r}".format(k)
                )
        self.entry_timestamp = entry_timestamp
        self.entry_index = int(entry_index)
        self.entry_price = float(entry_price)
        self.position_size = float(position_size)
        self.entry_signal_meta = dict(entry_signal_meta)


def _zero_costs():
    return {"fees": 0.0, "slippage": 0.0, "spread_cost": 0.0}


def complete_trade(
    position,
    current_candle,
    current_index,
    exit_decision,
    cost_model=None,
):
    """Build the canonical completed-trade record extended with Phase-2 exit
    fields.

    Args:
      position:        OpenPosition.
      current_candle:  the present candle (must contain 'timestamp' and
                       'close').
      current_index:   present index (>= position.entry_index).
      exit_decision:   dict from `decide_exit` with should_exit=True.
      cost_model:      optional `hermes.safety.cost_model.CostModel`. When
                       provided, fees / slippage / spread_cost / net_pnl
                       are computed and the trade's `pnl` (and outcome)
                       reflect the post-cost result. When omitted, costs
                       are zero and net_pnl == gross pnl.

    Returns:
      A dict that satisfies `is_complete_trade_record` (Phase-1 contract)
      AND contains every field in REQUIRED_EXIT_FIELDS.
    """
    if not isinstance(position, OpenPosition):
        raise ValueError("position must be an OpenPosition instance")
    if not isinstance(current_candle, dict):
        raise ValueError("current_candle must be a dict")
    for k in ("timestamp", "close"):
        if k not in current_candle:
            raise ValueError(
                "current_candle missing required key: {!r}".format(k)
            )
    if (
        not isinstance(current_index, int)
        or isinstance(current_index, bool)
        or current_index < 0
    ):
        raise ValueError("current_index must be a non-negative int")
    if current_index < position.entry_index:
        raise ValueError("current_index must be >= position.entry_index")
    if not isinstance(exit_decision, dict):
        raise ValueError("exit_decision must be a dict")
    if not exit_decision.get("should_exit"):
        raise ValueError(
            "complete_trade called for a non-exiting decision"
        )
    if "exit_reason" not in exit_decision:
        raise ValueError("exit_decision missing 'exit_reason'")

    # Lazy import to avoid circular dependency with cost_model module.
    if cost_model is not None:
        from hermes.safety.cost_model import CostModel

        if not isinstance(cost_model, CostModel):
            raise ValueError("cost_model must be a CostModel instance")

    exit_price = float(exit_decision.get("exit_price", current_candle["close"]))
    bars_held = int(exit_decision.get(
        "bars_held", current_index - position.entry_index
    ))

    gross_pnl = (
        (exit_price - position.entry_price) * position.position_size
    )

    if cost_model is not None:
        breakdown = cost_model.apply(
            position.entry_price, exit_price, position.position_size
        )
        fees = breakdown["fees"]
        slippage = breakdown["slippage"]
        spread_cost = breakdown["spread_cost"]
        net_pnl = breakdown["net_pnl"]
    else:
        zeros = _zero_costs()
        fees = zeros["fees"]
        slippage = zeros["slippage"]
        spread_cost = zeros["spread_cost"]
        net_pnl = gross_pnl

    meta = position.entry_signal_meta
    entry = {
        "timestamp": position.entry_timestamp,
        "date": meta["date"],
        "hour": meta["hour"],
        "entry_price": position.entry_price,
        "sequence_value": meta["sequence_value"],
        "amd_value": meta["amd_value"],
        "combined_value": meta["combined_value"],
        "agreement": meta["agreement"],
        "confidence": meta["confidence"],
        "regime": meta["regime"],
        "momentum_score": meta["momentum_score"],
        "volatility_score": meta["volatility_score"],
    }
    # Use net_pnl as the trade's canonical pnl so outcome reflects cost
    # reality. The Phase-1 builder will derive outcome from pnl.
    exit_data = {
        "exit_timestamp": current_candle["timestamp"],
        "exit_price": exit_price,
        "pnl": float(net_pnl),
        "net_pnl": float(net_pnl),
    }
    record = build_completed_trade(entry, exit_data)

    # Extend additively with Phase-2 exit fields.
    record["exit_reason"] = exit_decision["exit_reason"]
    record["bars_held"] = int(bars_held)
    record["fees"] = float(fees)
    record["slippage"] = float(slippage)
    record["spread_cost"] = float(spread_cost)
    record["gross_pnl"] = float(gross_pnl)

    # Defense-in-depth: confirm the canonical contract still holds.
    if not is_complete_trade_record(record):
        raise ValueError(
            "lifecycle produced a record that fails Phase-1 validation"
        )
    # And confirm Phase-2 exit fields are all present.
    for k in REQUIRED_EXIT_FIELDS:
        if k not in record:
            raise ValueError(
                "completed trade missing exit field: {!r}".format(k)
            )
    return record


__all__ = [
    "OpenPosition",
    "REQUIRED_EXIT_FIELDS",
    "REQUIRED_OPEN_POSITION_META_KEYS",
    "REQUIRED_TRADE_KEYS",
    "complete_trade",
    "is_complete_trade_record",
]
