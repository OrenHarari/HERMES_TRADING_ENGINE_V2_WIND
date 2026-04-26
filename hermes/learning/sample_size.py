"""Prompt 2 / Step 6C - Sample Size Hardening.

Stand-alone wrapper that:
  - centralizes the configurable minimums (matching the spec defaults)
  - exposes a `check_sample_size(total_trades, config)` pure function
  - provides `adapt_thresholds_safely(adapter, ...)` which canonicalizes the
    Phase-2 spec reason "insufficient_sample_size" while preserving every
    output key returned by the Phase-1 `ThresholdAdapter.propose`.

The Phase-1 `ThresholdAdapter` is unchanged. Its own reason
"insufficient_total_trades" remains valid for direct callers; the Step-6C
wrapper canonicalizes that to "insufficient_sample_size" per Prompt 2.

Bucket / regime / combination minimums are enforced inside
`hermes.learning.attribution.compute_attribution` and have always defaulted
to the spec values. This module re-exports those defaults so callers have
one place to override them.
"""

from hermes.learning.attribution import AttributionConfig

DEFAULT_MIN_TRADES_TOTAL = 100
DEFAULT_MIN_TRADES_PER_BUCKET = 20
DEFAULT_MIN_TRADES_PER_REGIME = 30
DEFAULT_MIN_TRADES_PER_COMBINATION = 50

REASON_INSUFFICIENT_SAMPLE_SIZE = "insufficient_sample_size"


class SampleSizeConfig(object):
    """Configurable minimums for sample-size hardening.

    Defaults match the Prompt 2 / Step 6C spec exactly. Tests may inject
    smaller values; production callers should keep these defaults.
    """

    __slots__ = (
        "min_trades_total",
        "min_trades_per_bucket",
        "min_trades_per_regime",
        "min_trades_per_combination",
    )

    def __init__(
        self,
        min_trades_total=DEFAULT_MIN_TRADES_TOTAL,
        min_trades_per_bucket=DEFAULT_MIN_TRADES_PER_BUCKET,
        min_trades_per_regime=DEFAULT_MIN_TRADES_PER_REGIME,
        min_trades_per_combination=DEFAULT_MIN_TRADES_PER_COMBINATION,
    ):
        for name, val in (
            ("min_trades_total", min_trades_total),
            ("min_trades_per_bucket", min_trades_per_bucket),
            ("min_trades_per_regime", min_trades_per_regime),
            ("min_trades_per_combination", min_trades_per_combination),
        ):
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ValueError("{!s} must be int >= 0".format(name))
        self.min_trades_total = min_trades_total
        self.min_trades_per_bucket = min_trades_per_bucket
        self.min_trades_per_regime = min_trades_per_regime
        self.min_trades_per_combination = min_trades_per_combination

    def to_attribution_config(self):
        """Project this config onto an `AttributionConfig`."""
        return AttributionConfig(
            min_trades_per_bucket=self.min_trades_per_bucket,
            min_trades_per_regime=self.min_trades_per_regime,
            min_trades_per_combination=self.min_trades_per_combination,
        )


def check_sample_size(total_trades, config=None):
    """Return {"sample_size_ok": bool, "reason": str, "details": dict}."""
    if config is None:
        config = SampleSizeConfig()
    if not isinstance(total_trades, int) or isinstance(total_trades, bool):
        raise ValueError("total_trades must be int")
    if total_trades < 0:
        raise ValueError("total_trades must be >= 0")
    if total_trades < config.min_trades_total:
        return {
            "sample_size_ok": False,
            "reason": REASON_INSUFFICIENT_SAMPLE_SIZE,
            "details": {
                "total_trades": total_trades,
                "required": config.min_trades_total,
            },
        }
    return {
        "sample_size_ok": True,
        "reason": "",
        "details": {
            "total_trades": total_trades,
            "required": config.min_trades_total,
        },
    }


def adapt_thresholds_safely(
    adapter,
    attribution_result,
    total_trades,
    active_thresholds,
    sample_size_config=None,
):
    """Phase-2 canonical wrapper around `ThresholdAdapter.propose`.

    If sample size is insufficient, returns a dict with the spec-canonical
    reason "insufficient_sample_size" while preserving every Phase-1 output
    key. Otherwise delegates to the adapter unchanged.
    """
    # Lazy import to avoid circular references at module-load time.
    from hermes.learning.threshold_adapter import ThresholdAdapter

    if not isinstance(adapter, ThresholdAdapter):
        raise ValueError("adapter must be a ThresholdAdapter instance")
    if not isinstance(active_thresholds, dict):
        raise ValueError("active_thresholds must be a dict")
    ss = check_sample_size(total_trades, config=sample_size_config)
    if not ss["sample_size_ok"]:
        return {
            "thresholds_adapted": False,
            "candidate_thresholds_created": False,
            "reason": REASON_INSUFFICIENT_SAMPLE_SIZE,
            "active_thresholds_after": dict(active_thresholds),
            "proposals": {},
            "log": [
                {
                    "decision": "no_adaptation_sample_size_gate",
                    "reason": REASON_INSUFFICIENT_SAMPLE_SIZE,
                    "details": ss["details"],
                }
            ],
        }
    return adapter.propose(attribution_result, total_trades, active_thresholds)


__all__ = [
    "DEFAULT_MIN_TRADES_PER_BUCKET",
    "DEFAULT_MIN_TRADES_PER_COMBINATION",
    "DEFAULT_MIN_TRADES_PER_REGIME",
    "DEFAULT_MIN_TRADES_TOTAL",
    "REASON_INSUFFICIENT_SAMPLE_SIZE",
    "SampleSizeConfig",
    "adapt_thresholds_safely",
    "check_sample_size",
]
