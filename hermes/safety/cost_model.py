"""Prompt 2 / Step 5C - Cost, Slippage, and Execution Model.

Stand-alone, deterministic, configurable, explainable cost model.

  CostModel(fee_pct, slippage_pct, spread_pct,
            entry_fill_model, exit_fill_model)

  cost_model.apply(entry_price, exit_price, position_size) ->
      { "gross_pnl", "fees", "slippage", "spread_cost", "net_pnl",
        "entry_fill_model", "exit_fill_model" }

Validation gates:

  validate_cost_model(cost_model) -> {"validation_passed", "reason"}
  check_cost_model_for_live(cost_model) -> {"trade_allowed", "reason"}

Helper:

  apply_cost_model_to_trade(trade, cost_model) -> trade copy with cost fields
  populated.

Backward compatibility:
- Phase 1 long-only convention: gross_pnl = (exit - entry) * size.
- Costs are non-negative additive deductions; net_pnl = gross - fees -
  slippage - spread_cost.
- Costs are computed off the round-trip notional (entry+exit)*size, which
  captures both-side fees/slippage in one expression and is the simplest
  determ. choice.
"""

from hermes.utils.bounds import is_real_number

REASON_OK = ""
REASON_MISSING_COST_MODEL = "missing_cost_model"

VALID_FILL_MODELS = ("quoted", "with_slippage")

# Sanity ceiling on per-side cost components: more than 100% of notional is
# almost certainly a config error. Tighter than reality but still very loose.
_MAX_PCT = 1.0


def _check_pct(name, value):
    if not is_real_number(value):
        raise ValueError("{!s} must be numeric".format(name))
    if value < 0.0:
        raise ValueError("{!s} must be >= 0".format(name))
    if value > _MAX_PCT:
        raise ValueError("{!s} must be <= 1.0".format(name))


class CostModel(object):
    """Deterministic cost model. All components in [0, 1] (fraction of notional).
    """

    __slots__ = (
        "fee_pct",
        "slippage_pct",
        "spread_pct",
        "entry_fill_model",
        "exit_fill_model",
    )

    def __init__(
        self,
        fee_pct=0.0,
        slippage_pct=0.0,
        spread_pct=0.0,
        entry_fill_model="with_slippage",
        exit_fill_model="with_slippage",
    ):
        _check_pct("fee_pct", fee_pct)
        _check_pct("slippage_pct", slippage_pct)
        _check_pct("spread_pct", spread_pct)
        if entry_fill_model not in VALID_FILL_MODELS:
            raise ValueError(
                "entry_fill_model must be one of {!s}".format(VALID_FILL_MODELS)
            )
        if exit_fill_model not in VALID_FILL_MODELS:
            raise ValueError(
                "exit_fill_model must be one of {!s}".format(VALID_FILL_MODELS)
            )
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.spread_pct = float(spread_pct)
        self.entry_fill_model = entry_fill_model
        self.exit_fill_model = exit_fill_model

    def apply(self, entry_price, exit_price, position_size):
        """Return cost breakdown for one completed long trade."""
        if not is_real_number(entry_price) or entry_price <= 0.0:
            raise ValueError("entry_price must be > 0")
        if not is_real_number(exit_price) or exit_price <= 0.0:
            raise ValueError("exit_price must be > 0")
        if not is_real_number(position_size) or position_size < 0.0:
            raise ValueError("position_size must be >= 0")

        notional = (float(entry_price) + float(exit_price)) * float(position_size)
        fees = self.fee_pct * notional
        slippage = self.slippage_pct * notional
        spread_cost = self.spread_pct * notional
        gross_pnl = (float(exit_price) - float(entry_price)) * float(position_size)
        net_pnl = gross_pnl - fees - slippage - spread_cost
        return {
            "gross_pnl": float(gross_pnl),
            "fees": float(fees),
            "slippage": float(slippage),
            "spread_cost": float(spread_cost),
            "net_pnl": float(net_pnl),
            "entry_fill_model": self.entry_fill_model,
            "exit_fill_model": self.exit_fill_model,
        }


def validate_cost_model(cost_model):
    """Validation-time gate: returns {"validation_passed", "reason"}."""
    if cost_model is None or not isinstance(cost_model, CostModel):
        return {"validation_passed": False, "reason": REASON_MISSING_COST_MODEL}
    return {"validation_passed": True, "reason": REASON_OK}


def check_cost_model_for_live(cost_model):
    """Live-mode gate: returns {"trade_allowed", "reason"}.

    In live_mode, missing cost model must block trading per Prompt 2 spec.
    """
    if cost_model is None or not isinstance(cost_model, CostModel):
        return {"trade_allowed": False, "reason": REASON_MISSING_COST_MODEL}
    return {"trade_allowed": True, "reason": REASON_OK}


def apply_cost_model_to_trade(trade, cost_model):
    """Return a NEW trade dict with cost fields populated.

    `trade` must contain entry_price, exit_price, position_size. The original
    dict is not mutated.

    The output dict carries:
      - all original fields
      - gross_pnl, fees, slippage, spread_cost, net_pnl
      - entry_fill_model, exit_fill_model
    """
    if not isinstance(trade, dict):
        raise ValueError("trade must be a dict")
    for k in ("entry_price", "exit_price", "position_size"):
        if k not in trade:
            raise ValueError(
                "trade missing required field for cost model: {!r}".format(k)
            )
    if not isinstance(cost_model, CostModel):
        raise ValueError("cost_model must be a CostModel instance")

    breakdown = cost_model.apply(
        trade["entry_price"], trade["exit_price"], trade["position_size"]
    )
    out = dict(trade)
    out.update(breakdown)
    return out
