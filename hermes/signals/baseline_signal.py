"""Phase 3A - baseline signal (engineering scaffolding ONLY).

!!! NOT A REAL TRADING EDGE !!!

This module exists solely so the offline backtest end-to-end pipeline
can be exercised against OHLCV-only CSVs that do not carry the raw
`sequence_value`, `amd_value`, `combined_value` columns.

The formula is a deterministic, stateless transformation of past closes:

    N = BASELINE_LOOKBACK = 10 bars
    trend          = (close[i] - close[i - N + 1]) / close[i - N + 1]
    sequence_value = clamp01(0.5 + trend * 5.0)
    amd_value      = clamp01(min(1.0, abs(trend) * 50.0))
    combined_value = clamp01((sequence_value + amd_value) / 2.0)

If insufficient history is present (i + 1 < N), the neutral triple
(0.5, 0.5, 0.5) is returned. The function NEVER reads `candles[i+1:]`.

Replace this with a real signal generator in a later phase before
making any profitability claim.

Imported directly via `hermes.signals.baseline_signal`. The signals
package `__init__.py` is intentionally NOT modified by Phase 3A.
"""

BASELINE_LOOKBACK = 10
NEUTRAL_TRIPLE = {
    "sequence_value": 0.5,
    "amd_value": 0.5,
    "combined_value": 0.5,
}


def _clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def baseline_signal(candles, current_index):
    """Return a synthetic signal triple from past closes ONLY.

    This is engineering scaffolding, not a trading edge.

    Parameters
    ----------
    candles : list of dict
        Each candle must have a numeric `close`.
    current_index : int
        Index of the bar at which the signal is being generated. Only
        candles at indices `[0 .. current_index]` are read.

    Returns
    -------
    dict with keys: sequence_value, amd_value, combined_value
        Each value is a float in [0.0, 1.0]. Output is deterministic
        for any given (candles, current_index).
    """
    if not isinstance(candles, list):
        raise ValueError("candles must be a list")
    if not isinstance(current_index, int):
        raise ValueError("current_index must be an int")
    if current_index < 0 or current_index >= len(candles):
        raise ValueError(
            "current_index {0} out of range for candles of length {1}".format(
                current_index, len(candles)
            )
        )

    # Insufficient history -> return neutral triple.
    if current_index + 1 < BASELINE_LOOKBACK:
        return dict(NEUTRAL_TRIPLE)

    start_close = candles[current_index - BASELINE_LOOKBACK + 1]["close"]
    end_close = candles[current_index]["close"]
    if not isinstance(start_close, (int, float)) or start_close <= 0:
        # Defensive: refuse to divide by non-positive base.
        return dict(NEUTRAL_TRIPLE)

    trend = (float(end_close) - float(start_close)) / float(start_close)
    abs_trend = abs(trend)

    sequence_value = _clamp01(0.5 + trend * 5.0)
    amd_value = _clamp01(min(1.0, abs_trend * 50.0))
    combined_value = _clamp01((sequence_value + amd_value) / 2.0)

    return {
        "sequence_value": sequence_value,
        "amd_value": amd_value,
        "combined_value": combined_value,
    }


__all__ = ["BASELINE_LOOKBACK", "NEUTRAL_TRIPLE", "baseline_signal"]
