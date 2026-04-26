"""Step 4 - Market intelligence assembly.

Single entry point that returns the canonical market-context dict for the
decision/learning layers.

Output keys (exact set):
  - regime:           one of REGIME_VALUES
  - volatility_score: float in [0, 1]
  - momentum_score:   float in [0, 1]

Volume confirmation is optional: if volume keys are present in candles, this
module does not currently change the regime, but records nothing fabricated.
Missing volume MUST NOT fail the call.
"""

from hermes.market.config import MarketIntelligenceConfig
from hermes.market.momentum import compute_momentum_score
from hermes.market.regime import (
    REGIME_CHOP,
    REGIME_HIGH_VOLATILITY,
    REGIME_LOW_VOLATILITY,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    classify_regime,
)
from hermes.market.volatility import compute_volatility_score

REGIME_VALUES = (
    REGIME_TREND_UP,
    REGIME_TREND_DOWN,
    REGIME_CHOP,
    REGIME_HIGH_VOLATILITY,
    REGIME_LOW_VOLATILITY,
)

REQUIRED_INTELLIGENCE_KEYS = ("regime", "volatility_score", "momentum_score")


def assemble_intelligence(candles, current_index, config=None):
    """Compute market intelligence at `current_index` using only past+present.

    Returns a dict with keys REQUIRED_INTELLIGENCE_KEYS.
    Pure function. Same input -> same output.
    """
    if config is None:
        config = MarketIntelligenceConfig()
    volatility_score = compute_volatility_score(candles, current_index, config)
    momentum_score = compute_momentum_score(candles, current_index, config)
    regime = classify_regime(volatility_score, momentum_score, config)
    return {
        "regime": regime,
        "volatility_score": volatility_score,
        "momentum_score": momentum_score,
    }
