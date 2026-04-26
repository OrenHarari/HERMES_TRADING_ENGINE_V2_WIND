"""Market intelligence layer (Prompt 1, Step 4).

Encapsulates price-only context analysis: regime classification, volatility
estimation, and momentum confirmation. Volume is used if available; absence
is tolerated (never invented).

All functions in this package take `(candles, current_index)` and use only
candles with index <= current_index. Future candles are NEVER read.
"""

from hermes.market.config import MarketIntelligenceConfig
from hermes.market.intelligence import (
    REGIME_VALUES,
    REQUIRED_INTELLIGENCE_KEYS,
    assemble_intelligence,
)
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

__all__ = [
    "MarketIntelligenceConfig",
    "REGIME_CHOP",
    "REGIME_HIGH_VOLATILITY",
    "REGIME_LOW_VOLATILITY",
    "REGIME_TREND_DOWN",
    "REGIME_TREND_UP",
    "REGIME_VALUES",
    "REQUIRED_INTELLIGENCE_KEYS",
    "assemble_intelligence",
    "classify_regime",
    "compute_momentum_score",
    "compute_volatility_score",
]
