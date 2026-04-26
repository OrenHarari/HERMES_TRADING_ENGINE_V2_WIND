"""Decision layer (Prompt 1, Step 5).

Contains the confidence model, eligibility gate, validation gate, performance
report, and the top-level make_decision() entry point.

The canonical decision output dict is:

  {
      "trade_allowed": bool,
      "confidence": float,
      "agreement": float,
      "regime": str,
      "position_size": float,
      "reason": str,
  }
"""

from hermes.decision.config import DecisionConfig
from hermes.decision.confidence import compute_confidence_score, default_regime_weights
from hermes.decision.decision import REQUIRED_DECISION_KEYS, make_decision
from hermes.decision.eligibility import check_eligibility
from hermes.decision.performance import (
    PERFORMANCE_REPORT_KEYS,
    compute_performance_report,
)
from hermes.decision.validation import (
    assert_deterministic_replay,
    assert_no_future_data,
    validate_backtest,
)

__all__ = [
    "DecisionConfig",
    "PERFORMANCE_REPORT_KEYS",
    "REQUIRED_DECISION_KEYS",
    "assert_deterministic_replay",
    "assert_no_future_data",
    "check_eligibility",
    "compute_confidence_score",
    "compute_performance_report",
    "default_regime_weights",
    "make_decision",
    "validate_backtest",
]
