"""Configuration for market intelligence (Step 4).

Pure data class. All thresholds explicit and deterministic.
"""


class MarketIntelligenceConfig(object):
    """Configurable thresholds for the market intelligence layer.

    Attributes:
      lookback                : window size for vol / momentum (default 20)
      volatility_cap          : relative range that maps to score=1.0 (default 0.05)
      momentum_cap            : abs relative change that maps to score=1.0 / 0.0
                                (default 0.05)
      high_volatility_threshold : volatility_score >= this -> high_volatility regime
      low_volatility_threshold  : volatility_score <= this -> low_volatility regime
      trend_up_threshold        : momentum_score >= this -> trend_up regime
      trend_down_threshold      : momentum_score <= this -> trend_down regime

    All thresholds are inclusive on the regime-defining side (>= for upper,
    <= for lower) so the classification function is total and deterministic
    at exact boundary values.
    """

    __slots__ = (
        "lookback",
        "volatility_cap",
        "momentum_cap",
        "high_volatility_threshold",
        "low_volatility_threshold",
        "trend_up_threshold",
        "trend_down_threshold",
    )

    def __init__(
        self,
        lookback=20,
        volatility_cap=0.05,
        momentum_cap=0.05,
        high_volatility_threshold=0.80,
        low_volatility_threshold=0.20,
        trend_up_threshold=0.65,
        trend_down_threshold=0.35,
    ):
        if not isinstance(lookback, int) or lookback < 2:
            raise ValueError("lookback must be int >= 2; got {!r}".format(lookback))
        if volatility_cap <= 0.0:
            raise ValueError(
                "volatility_cap must be > 0; got {!r}".format(volatility_cap)
            )
        if momentum_cap <= 0.0:
            raise ValueError(
                "momentum_cap must be > 0; got {!r}".format(momentum_cap)
            )
        if not (0.0 <= low_volatility_threshold < high_volatility_threshold <= 1.0):
            raise ValueError(
                "require 0 <= low_volatility_threshold < high_volatility_threshold <= 1"
            )
        if not (0.0 <= trend_down_threshold < trend_up_threshold <= 1.0):
            raise ValueError(
                "require 0 <= trend_down_threshold < trend_up_threshold <= 1"
            )
        self.lookback = lookback
        self.volatility_cap = float(volatility_cap)
        self.momentum_cap = float(momentum_cap)
        self.high_volatility_threshold = float(high_volatility_threshold)
        self.low_volatility_threshold = float(low_volatility_threshold)
        self.trend_up_threshold = float(trend_up_threshold)
        self.trend_down_threshold = float(trend_down_threshold)
