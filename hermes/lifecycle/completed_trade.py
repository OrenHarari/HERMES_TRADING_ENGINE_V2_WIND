"""Phase 1 minimal completed-trade builder.

Produces the canonical record shape that learning/memory.py will accept.
The required-keys list is the strict subset that Phase 2 Step 5B will extend
additively (with fees, slippage, spread_cost, bars_held, exit_reason).

A trade is COMPLETED iff it has both an entry and an exit. This module
does not log incomplete trades.
"""

from hermes.market import REGIME_VALUES
from hermes.utils.bounds import is_numeric, is_unit_interval

OUTCOME_WIN = "win"
OUTCOME_LOSS = "loss"
OUTCOME_BREAKEVEN = "breakeven"
OUTCOME_VALUES = (OUTCOME_WIN, OUTCOME_LOSS, OUTCOME_BREAKEVEN)

# The strict Phase 1 set. Phase 2 (5B) will add additional keys; new keys
# may be added but none of these may be removed or renamed.
REQUIRED_TRADE_KEYS = (
    "timestamp",        # entry timestamp (numeric)
    "date",             # 'YYYY-MM-DD' or any deterministic string
    "hour",             # int hour-of-day, 0..23
    "exit_timestamp",   # numeric, > timestamp
    "entry_price",
    "exit_price",
    "sequence_value",
    "amd_value",
    "combined_value",
    "agreement",
    "confidence",
    "regime",
    "momentum_score",
    "volatility_score",
    "outcome",
    "pnl",
)


def _outcome_from_pnl(pnl):
    if pnl > 0.0:
        return OUTCOME_WIN
    if pnl < 0.0:
        return OUTCOME_LOSS
    return OUTCOME_BREAKEVEN


def is_complete_trade_record(record):
    """Return True iff `record` has all REQUIRED_TRADE_KEYS with valid values.

    Does not raise.
    """
    try:
        _validate_record(record)
        return True
    except ValueError:
        return False


def _validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("trade record must be a dict")
    for k in REQUIRED_TRADE_KEYS:
        if k not in record:
            raise ValueError("trade record missing required key: {!r}".format(k))

    if not is_numeric(record["timestamp"]):
        raise ValueError("timestamp must be numeric")
    if not is_numeric(record["exit_timestamp"]):
        raise ValueError("exit_timestamp must be numeric")
    if record["exit_timestamp"] < record["timestamp"]:
        raise ValueError("exit_timestamp must be >= timestamp")
    if not isinstance(record["date"], str) or not record["date"]:
        raise ValueError("date must be a non-empty string")
    if (
        not isinstance(record["hour"], int)
        or isinstance(record["hour"], bool)
        or not (0 <= record["hour"] <= 23)
    ):
        raise ValueError("hour must be int in [0,23]")
    if not is_numeric(record["entry_price"]) or record["entry_price"] <= 0.0:
        raise ValueError("entry_price must be a positive number")
    if not is_numeric(record["exit_price"]) or record["exit_price"] <= 0.0:
        raise ValueError("exit_price must be a positive number")
    for k in (
        "sequence_value",
        "amd_value",
        "combined_value",
        "agreement",
        "confidence",
        "momentum_score",
        "volatility_score",
    ):
        if not is_unit_interval(record[k]):
            raise ValueError("{!s} must be a real number in [0,1]".format(k))
    if record["regime"] not in REGIME_VALUES:
        raise ValueError("regime must be one of {!s}".format(REGIME_VALUES))
    if record["outcome"] not in OUTCOME_VALUES:
        raise ValueError("outcome must be one of {!s}".format(OUTCOME_VALUES))
    if not is_numeric(record["pnl"]):
        raise ValueError("pnl must be numeric")
    if "net_pnl" in record and not is_numeric(record["net_pnl"]):
        raise ValueError("net_pnl must be numeric when present")


def build_completed_trade(entry, exit_data):
    """Build a canonical completed-trade record from entry + exit information.

    `entry` (dict) must contain:
      - timestamp, date, hour
      - entry_price
      - sequence_value, amd_value, combined_value, agreement
      - confidence, regime, momentum_score, volatility_score

    `exit_data` (dict) must contain:
      - exit_timestamp
      - exit_price
      - pnl
      - net_pnl (optional)
      - notes  (optional)

    Returns a dict with all REQUIRED_TRADE_KEYS plus optional 'net_pnl' and
    'notes'. The outcome is derived deterministically from pnl.

    Raises ValueError if any required input is missing or invalid.
    """
    if not isinstance(entry, dict):
        raise ValueError("entry must be a dict")
    if not isinstance(exit_data, dict):
        raise ValueError("exit_data must be a dict")

    record = {
        "timestamp": entry.get("timestamp"),
        "date": entry.get("date"),
        "hour": entry.get("hour"),
        "exit_timestamp": exit_data.get("exit_timestamp"),
        "entry_price": entry.get("entry_price"),
        "exit_price": exit_data.get("exit_price"),
        "sequence_value": entry.get("sequence_value"),
        "amd_value": entry.get("amd_value"),
        "combined_value": entry.get("combined_value"),
        "agreement": entry.get("agreement"),
        "confidence": entry.get("confidence"),
        "regime": entry.get("regime"),
        "momentum_score": entry.get("momentum_score"),
        "volatility_score": entry.get("volatility_score"),
        "pnl": exit_data.get("pnl"),
    }
    pnl = record["pnl"]
    if not is_numeric(pnl):
        raise ValueError("pnl must be numeric")
    record["outcome"] = _outcome_from_pnl(pnl)
    if "net_pnl" in exit_data:
        record["net_pnl"] = exit_data["net_pnl"]
    if "notes" in exit_data:
        record["notes"] = exit_data["notes"]
    _validate_record(record)
    return record
