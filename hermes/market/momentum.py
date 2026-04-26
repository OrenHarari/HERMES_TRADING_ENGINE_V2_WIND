"""Step 4 - Momentum confirmation.

Direction-aware momentum mapped into [0, 1] with 0.5 = neutral.
Uses only candles with index <= current_index.
"""

from hermes.market.config import MarketIntelligenceConfig
from hermes.market.volatility import _candle_get, _slice_window, _validate_candles


def compute_momentum_score(candles, current_index, config=None):
    """Return momentum_score in [0, 1] for the window ending at current_index.

    Definition:
      first_close = window[0].close
      last_close  = window[-1].close
      rel_change  = (last_close - first_close) / first_close
      raw         = rel_change / momentum_cap   # in [-inf, +inf]
      score       = clamp((raw + 1.0) / 2.0, 0.0, 1.0)

    score = 0.5 -> neutral momentum
    score >= trend_up_threshold (default 0.65) suggests trend_up
    score <= trend_down_threshold (default 0.35) suggests trend_down

    Single-candle windows or zero/negative first_close return 0.5 (neutral),
    since there is no informative direction.
    """
    if config is None:
        config = MarketIntelligenceConfig()
    _validate_candles(candles, current_index)
    window = _slice_window(candles, current_index, config.lookback)
    if len(window) < 2:
        return 0.5
    first_close = _candle_get(window[0], "close")
    last_close = _candle_get(window[-1], "close")
    if first_close <= 0.0:
        return 0.5
    rel_change = (last_close - first_close) / first_close
    raw = rel_change / config.momentum_cap
    score = (raw + 1.0) / 2.0
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return float(score)
