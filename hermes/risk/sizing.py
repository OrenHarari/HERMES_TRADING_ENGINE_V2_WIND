"""Step 5 / Part 3 - Baseline position sizing.

Phase 1 baseline only. Position size is `base * confidence_multiplier`
where the multiplier is derived from confidence and capped.

Phase 2 (Step 5D) will wrap this with stop-distance, equity, and
max_position_pct_of_equity caps in `hermes/risk/sizing_safety.py` - those caps
are NOT implemented here by design.
"""

from hermes.utils.bounds import is_real_number, is_unit_interval


def compute_position_size(base_size, confidence, confidence_multiplier_cap=1.0):
    """Return a non-negative position size.

    Args:
      base_size: numeric baseline size (>= 0).
      confidence: float in [0, 1].
      confidence_multiplier_cap: float > 0; caps how much confidence can scale.

    Returns:
      float position size in [0, base_size * confidence_multiplier_cap].
    """
    if not is_real_number(base_size):
        raise ValueError("base_size must be numeric")
    if base_size < 0.0:
        raise ValueError("base_size must be >= 0")
    if not is_unit_interval(confidence):
        raise ValueError("confidence must be in [0, 1]")
    if not is_real_number(confidence_multiplier_cap):
        raise ValueError("confidence_multiplier_cap must be numeric")
    if confidence_multiplier_cap <= 0.0:
        raise ValueError("confidence_multiplier_cap must be > 0")

    multiplier = float(confidence)
    cap = float(confidence_multiplier_cap)
    if multiplier > cap:
        multiplier = cap
    return float(base_size) * multiplier
