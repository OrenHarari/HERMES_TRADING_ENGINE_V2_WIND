"""Step 4 - Market regime classification.

Deterministic rule-based classifier producing exactly one of:
  trend_up, trend_down, chop, high_volatility, low_volatility
"""

from hermes.market.config import MarketIntelligenceConfig

REGIME_TREND_UP = "trend_up"
REGIME_TREND_DOWN = "trend_down"
REGIME_CHOP = "chop"
REGIME_HIGH_VOLATILITY = "high_volatility"
REGIME_LOW_VOLATILITY = "low_volatility"


def classify_regime(volatility_score, momentum_score, config=None):
    """Classify into one regime from value-only thresholds.

    Order of checks (first match wins):
      1. volatility_score >= high_volatility_threshold -> high_volatility
      2. volatility_score <= low_volatility_threshold  -> low_volatility
      3. momentum_score   >= trend_up_threshold        -> trend_up
      4. momentum_score   <= trend_down_threshold      -> trend_down
      5. otherwise                                      -> chop

    Pure function. Same input -> same output.
    """
    if config is None:
        config = MarketIntelligenceConfig()
    if isinstance(volatility_score, bool) or not isinstance(
        volatility_score, (int, float)
    ):
        raise ValueError("volatility_score must be numeric")
    if isinstance(momentum_score, bool) or not isinstance(
        momentum_score, (int, float)
    ):
        raise ValueError("momentum_score must be numeric")
    v = float(volatility_score)
    m = float(momentum_score)
    if v != v or m != m:
        raise ValueError("scores must not be NaN")
    if not (0.0 <= v <= 1.0):
        raise ValueError("volatility_score must be in [0,1]; got {!r}".format(v))
    if not (0.0 <= m <= 1.0):
        raise ValueError("momentum_score must be in [0,1]; got {!r}".format(m))

    if v >= config.high_volatility_threshold:
        return REGIME_HIGH_VOLATILITY
    if v <= config.low_volatility_threshold:
        return REGIME_LOW_VOLATILITY
    if m >= config.trend_up_threshold:
        return REGIME_TREND_UP
    if m <= config.trend_down_threshold:
        return REGIME_TREND_DOWN
    return REGIME_CHOP
