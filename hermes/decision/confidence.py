"""Step 5 / Part 1 - Confidence model.

Bounded, deterministic, explainable weighted blend over signal-layer and
market-intelligence outputs. No outcome leakage. No randomness.

Default formula (all inputs in [0, 1]):

  signal_strength    = (sequence_value + amd_value + combined_value) / 3
  volatility_factor  = 1 - volatility_score
  regime_weight      = DEFAULT_REGIME_WEIGHTS[regime]

  confidence = w_signal      * signal_strength
             + w_agreement   * agreement
             + w_momentum    * momentum_score
             + w_volatility  * volatility_factor
             + w_regime      * regime_weight

Default weights sum to 1.0 so the result is in [0, 1] without re-normalization.

Note: Phase 1 is long-only by spec, so momentum_score is used directly:
high momentum -> higher long confidence; low momentum -> lower long confidence.
"""

from hermes.market import (
    REGIME_CHOP,
    REGIME_HIGH_VOLATILITY,
    REGIME_LOW_VOLATILITY,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    REGIME_VALUES,
)
from hermes.utils.bounds import clip01, is_unit_interval

DEFAULT_WEIGHTS = {
    "signal": 0.30,
    "agreement": 0.20,
    "momentum": 0.20,
    "volatility": 0.15,
    "regime": 0.15,
}

# Long-only regime weights: trend_up favored, trend_down disfavored.
DEFAULT_REGIME_WEIGHTS = {
    REGIME_TREND_UP: 1.0,
    REGIME_TREND_DOWN: 0.0,
    REGIME_LOW_VOLATILITY: 0.7,
    REGIME_CHOP: 0.3,
    REGIME_HIGH_VOLATILITY: 0.3,
}


def default_regime_weights():
    """Return a fresh copy of the default regime weights dict."""
    return dict(DEFAULT_REGIME_WEIGHTS)


def _require_unit(name, value):
    if not is_unit_interval(value):
        raise ValueError(
            "{!s} must be a real number in [0,1]; got {!r}".format(name, value)
        )


def compute_confidence_score(
    signal_output, intelligence, regime_weights=None, weights=None
):
    """Return a deterministic confidence_score in [0, 1].

    `signal_output` must be a dict with sequence_value, amd_value,
    combined_value, agreement (e.g. orchestrator.build_signal_output result).

    `intelligence` must be a dict with regime, volatility_score, momentum_score
    (e.g. market.assemble_intelligence result).

    `regime_weights` defaults to DEFAULT_REGIME_WEIGHTS; missing regimes raise.
    `weights` defaults to DEFAULT_WEIGHTS; if provided, must sum to 1.0.
    """
    if not isinstance(signal_output, dict):
        raise ValueError("signal_output must be a dict")
    if not isinstance(intelligence, dict):
        raise ValueError("intelligence must be a dict")

    for k in ("sequence_value", "amd_value", "combined_value", "agreement"):
        if k not in signal_output:
            raise ValueError("signal_output missing key: {!r}".format(k))
        _require_unit(k, signal_output[k])

    for k in ("regime", "volatility_score", "momentum_score"):
        if k not in intelligence:
            raise ValueError("intelligence missing key: {!r}".format(k))
    _require_unit("volatility_score", intelligence["volatility_score"])
    _require_unit("momentum_score", intelligence["momentum_score"])
    if intelligence["regime"] not in REGIME_VALUES:
        raise ValueError("invalid regime: {!r}".format(intelligence["regime"]))

    if regime_weights is None:
        regime_weights = DEFAULT_REGIME_WEIGHTS
    if intelligence["regime"] not in regime_weights:
        raise ValueError(
            "regime_weights missing entry for {!r}".format(intelligence["regime"])
        )
    rw = regime_weights[intelligence["regime"]]
    _require_unit("regime_weight", rw)

    if weights is None:
        weights = DEFAULT_WEIGHTS
    expected_keys = {"signal", "agreement", "momentum", "volatility", "regime"}
    if set(weights.keys()) != expected_keys:
        raise ValueError(
            "weights must have keys {!s}; got {!s}".format(
                sorted(expected_keys), sorted(weights.keys())
            )
        )
    total_w = 0.0
    for k in expected_keys:
        _require_unit("weight[{}]".format(k), weights[k])
        total_w += float(weights[k])
    # Allow tiny float-noise around 1.0; reject anything noticeably off.
    if abs(total_w - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1.0; got {!r}".format(total_w))

    signal_strength = (
        float(signal_output["sequence_value"])
        + float(signal_output["amd_value"])
        + float(signal_output["combined_value"])
    ) / 3.0
    agreement = float(signal_output["agreement"])
    momentum_score = float(intelligence["momentum_score"])
    volatility_factor = 1.0 - float(intelligence["volatility_score"])

    score = (
        weights["signal"] * signal_strength
        + weights["agreement"] * agreement
        + weights["momentum"] * momentum_score
        + weights["volatility"] * volatility_factor
        + weights["regime"] * float(rw)
    )
    return float(clip01(score))
