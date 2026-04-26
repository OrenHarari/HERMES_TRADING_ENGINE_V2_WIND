"""Prompt 2 / Step 0 - Data Contract and Input Safety.

Stand-alone validators that block bad market data and unknown system modes
BEFORE any signal, regime, confidence, risk, or learning logic runs.

Public outputs:
  validate_system_mode(system_mode) -> {"trade_allowed": bool, "reason": str}
  validate_candle_schema(candles, current_index)
      -> {"trade_allowed": bool, "reason": str}
  validate_market_data(candles, current_index, system_mode=None,
                       now_ts=None, max_staleness_seconds=None)
      -> {"trade_allowed": bool, "reason": str}

Reasons (canonical strings; do not rename):
  ""                           - OK (trade_allowed=True)
  "invalid_system_mode"        - mode missing/unknown
  "invalid_market_data"        - schema, OHLC, sortedness, dedup, lookahead
  "stale_market_data"          - paper/live mode + last candle too old

Backward-compatibility: accepts system_mode=None (legacy Prompt 1 callers
without explicit mode). For legacy callers staleness checks are skipped.
This module does NOT modify any existing decision/learning code.
"""

from hermes.utils.bounds import is_numeric

# --- Canonical strings ----------------------------------------------------
REASON_OK = ""
REASON_INVALID_DATA = "invalid_market_data"
REASON_STALE_DATA = "stale_market_data"
REASON_INVALID_MODE = "invalid_system_mode"

SYSTEM_MODE_BACKTEST = "backtest_mode"
SYSTEM_MODE_PAPER = "paper_mode"
SYSTEM_MODE_LIVE = "live_mode"
VALID_SYSTEM_MODES = (SYSTEM_MODE_BACKTEST, SYSTEM_MODE_PAPER, SYSTEM_MODE_LIVE)

REQUIRED_CANDLE_FIELDS = ("timestamp", "open", "high", "low", "close")
_OHLC_FIELDS = ("open", "high", "low", "close")


def _ok():
    return {"trade_allowed": True, "reason": REASON_OK}


def _block(reason):
    return {"trade_allowed": False, "reason": reason}


# --- System mode ----------------------------------------------------------
def validate_system_mode(system_mode):
    """Return {"trade_allowed", "reason"} for a system_mode value.

    Accepts:
      - None (legacy Prompt 1 callers; treated as backtest semantics)
      - one of VALID_SYSTEM_MODES

    Rejects everything else with reason="invalid_system_mode".
    """
    if system_mode is None:
        return _ok()
    if not isinstance(system_mode, str) or not system_mode:
        return _block(REASON_INVALID_MODE)
    if system_mode not in VALID_SYSTEM_MODES:
        return _block(REASON_INVALID_MODE)
    return _ok()


# --- Candle schema --------------------------------------------------------
def _is_valid_timestamp(ts):
    """Timestamps may be int or non-empty str. Booleans rejected."""
    if isinstance(ts, bool):
        return False
    if isinstance(ts, int):
        return True
    if isinstance(ts, str) and ts:
        return True
    return False


def _validate_one_candle(candle):
    """Return None on OK, or a reason string for the first violation."""
    if not isinstance(candle, dict):
        return REASON_INVALID_DATA
    for k in REQUIRED_CANDLE_FIELDS:
        if k not in candle:
            return REASON_INVALID_DATA
    if not _is_valid_timestamp(candle["timestamp"]):
        return REASON_INVALID_DATA
    o = candle["open"]
    h = candle["high"]
    l = candle["low"]
    c = candle["close"]
    for v in (o, h, l, c):
        if not is_numeric(v):
            return REASON_INVALID_DATA
    if h < l:
        return REASON_INVALID_DATA
    if h < o:
        return REASON_INVALID_DATA
    if h < c:
        return REASON_INVALID_DATA
    if l > o:
        return REASON_INVALID_DATA
    if l > c:
        return REASON_INVALID_DATA
    if "volume" in candle:
        v = candle["volume"]
        if not is_numeric(v):
            return REASON_INVALID_DATA
        if v < 0.0:
            return REASON_INVALID_DATA
    return None


def _timestamps_strictly_increasing(window):
    """Verify ascending order and no duplicates over the visible window.

    Mixed numeric/string timestamps are NOT comparable in a stable way; the
    spec allows both types but does not allow mixing within one stream.
    """
    if len(window) < 2:
        return True
    first_kind = type(window[0]["timestamp"])
    # Accept all-int or all-str streams; reject mixed.
    for c in window[1:]:
        if not isinstance(c["timestamp"], first_kind):
            return False
    for i in range(1, len(window)):
        prev = window[i - 1]["timestamp"]
        cur = window[i]["timestamp"]
        if cur <= prev:
            return False
    return True


def validate_candle_schema(candles, current_index):
    """Validate the candles[: current_index + 1] visible window.

    Future candles (index > current_index) are intentionally NOT inspected;
    the visible-window-only contract enforces the no-future-data rule by
    construction.
    """
    if not isinstance(candles, list) or len(candles) == 0:
        return _block(REASON_INVALID_DATA)
    if not isinstance(current_index, int) or isinstance(current_index, bool):
        return _block(REASON_INVALID_DATA)
    if current_index < 0 or current_index >= len(candles):
        return _block(REASON_INVALID_DATA)
    window = candles[: current_index + 1]
    for c in window:
        violation = _validate_one_candle(c)
        if violation is not None:
            return _block(violation)
    if not _timestamps_strictly_increasing(window):
        return _block(REASON_INVALID_DATA)
    return _ok()


# --- Market-data top-level (mode + schema + lookahead + staleness) -------
def _last_visible_timestamp(candles, current_index):
    return candles[current_index]["timestamp"]


def validate_market_data(
    candles,
    current_index,
    system_mode=None,
    now_ts=None,
    max_staleness_seconds=None,
):
    """Top-level data-contract validator.

    Precedence (deterministic):
      1. invalid_system_mode  (mode unknown)
      2. invalid_market_data  (schema / OHLC / order / dedup / lookahead)
      3. stale_market_data    (paper/live + numeric ts + exceeds threshold)
      4. ok

    Parameters:
      candles: list of candle dicts.
      current_index: index of the present candle within `candles`.
      system_mode: one of VALID_SYSTEM_MODES, or None for legacy callers
                   (treated as backtest for staleness purposes).
      now_ts: optional numeric "current time" reference. Used for the
              lookahead probe and for staleness in paper/live mode.
      max_staleness_seconds: numeric threshold used only in paper/live mode.

    Returns: {"trade_allowed": bool, "reason": str}.
    """
    mode_check = validate_system_mode(system_mode)
    if not mode_check["trade_allowed"]:
        return mode_check

    schema_check = validate_candle_schema(candles, current_index)
    if not schema_check["trade_allowed"]:
        return schema_check

    # Lookahead probe: if a "now" reference exists and the visible candle
    # is in the future of that reference, treat as invalid market data.
    last_ts = _last_visible_timestamp(candles, current_index)
    if now_ts is not None and is_numeric(now_ts) and is_numeric(last_ts):
        if last_ts > now_ts:
            return _block(REASON_INVALID_DATA)

    # Staleness check: only in paper/live mode AND only when both numeric
    # ts and a threshold are provided. Backtest and legacy (None) skip.
    if (
        system_mode in (SYSTEM_MODE_PAPER, SYSTEM_MODE_LIVE)
        and now_ts is not None
        and max_staleness_seconds is not None
        and is_numeric(now_ts)
        and is_numeric(last_ts)
        and is_numeric(max_staleness_seconds)
    ):
        age = now_ts - last_ts
        if age > max_staleness_seconds:
            return _block(REASON_STALE_DATA)

    return _ok()
