"""Configuration for the decision layer (Step 5)."""

from hermes.utils.bounds import is_real_number, is_unit_interval


class DecisionConfig(object):
    """Thresholds and parameters for the decision layer.

    Attributes:
      min_confidence:          required confidence to allow a trade
      min_agreement:           required agreement to allow a trade
      allow_chop:              if False, regime=="chop" blocks the trade
      volatility_min:          minimum acceptable volatility_score
      volatility_max:          maximum acceptable volatility_score
      base_position_size:      base sizing unit (Phase 1 baseline only;
                               Phase 2 wraps with equity / stop-distance caps)
      confidence_multiplier_cap: hard cap on confidence-driven scaling
    """

    __slots__ = (
        "min_confidence",
        "min_agreement",
        "allow_chop",
        "volatility_min",
        "volatility_max",
        "base_position_size",
        "confidence_multiplier_cap",
    )

    def __init__(
        self,
        min_confidence=0.60,
        min_agreement=0.60,
        allow_chop=False,
        volatility_min=0.10,
        volatility_max=0.85,
        base_position_size=1.0,
        confidence_multiplier_cap=1.0,
    ):
        if not is_unit_interval(min_confidence):
            raise ValueError("min_confidence must be in [0,1]")
        if not is_unit_interval(min_agreement):
            raise ValueError("min_agreement must be in [0,1]")
        if not isinstance(allow_chop, bool):
            raise ValueError("allow_chop must be a bool")
        if not is_unit_interval(volatility_min):
            raise ValueError("volatility_min must be in [0,1]")
        if not is_unit_interval(volatility_max):
            raise ValueError("volatility_max must be in [0,1]")
        if volatility_min > volatility_max:
            raise ValueError("volatility_min must be <= volatility_max")
        if not is_real_number(base_position_size):
            raise ValueError("base_position_size must be numeric")
        if base_position_size < 0.0:
            raise ValueError("base_position_size must be >= 0")
        if not is_real_number(confidence_multiplier_cap):
            raise ValueError("confidence_multiplier_cap must be numeric")
        if confidence_multiplier_cap <= 0.0:
            raise ValueError("confidence_multiplier_cap must be > 0")
        self.min_confidence = float(min_confidence)
        self.min_agreement = float(min_agreement)
        self.allow_chop = bool(allow_chop)
        self.volatility_min = float(volatility_min)
        self.volatility_max = float(volatility_max)
        self.base_position_size = float(base_position_size)
        self.confidence_multiplier_cap = float(confidence_multiplier_cap)
