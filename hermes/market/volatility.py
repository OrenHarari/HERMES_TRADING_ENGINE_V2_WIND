"""Step 4 - Volatility estimation.

Range-based volatility. Computed only from candles with index <=
`current_index`. Output is in [0, 1].
"""

from hermes.market.config import MarketIntelligenceConfig
from hermes.utils.bounds import clip01, is_numeric

REQUIRED_CANDLE_KEYS = ("open", "high", "low", "close")


def _validate_candles(candles, current_index):
    if not isinstance(candles, list):
        raise ValueError("candles must be a list")
    if not isinstance(current_index, int):
        raise ValueError("current_index must be int")
    n = len(candles)
    if n == 0:
        raise ValueError("candles must be non-empty")
    if current_index < 0 or current_index >= n:
        raise ValueError(
            "current_index {!s} out of range [0, {!s})".format(current_index, n)
        )


def _slice_window(candles, current_index, lookback):
    """Return only past+present candles (no future data)."""
    start = max(0, current_index - lookback + 1)
    return candles[start : current_index + 1]


def _candle_get(candle, key):
    if not isinstance(candle, dict):
        raise ValueError("candle must be a dict")
    if key not in candle:
        raise ValueError("candle missing required key: {!r}".format(key))
    v = candle[key]
    if not is_numeric(v):
        raise ValueError("candle[{!r}] must be numeric (non-bool, non-NaN); got {!r}".format(key, v))
    return float(v)


def compute_volatility_score(candles, current_index, config=None):
    """Return volatility_score in [0, 1] for the window ending at current_index.

    Definition:
      relative_range = (max(high) - min(low)) / mean(close)
      volatility_score = min(relative_range / volatility_cap, 1.0)

    Uses only candles[: current_index + 1]. Never reads future candles.
    """
    if config is None:
        config = MarketIntelligenceConfig()
    _validate_candles(candles, current_index)
    window = _slice_window(candles, current_index, config.lookback)
    highs = [_candle_get(c, "high") for c in window]
    lows = [_candle_get(c, "low") for c in window]
    closes = [_candle_get(c, "close") for c in window]
    mean_close = sum(closes) / len(closes)
    if mean_close <= 0.0:
        return 0.0
    relative_range = (max(highs) - min(lows)) / mean_close
    if relative_range < 0.0:
        relative_range = 0.0
    return float(clip01(relative_range / config.volatility_cap))
