"""Learning loop (Prompt 1, Step 6).

Sub-modules:
  - memory:            append-only completed-trade JSON store
  - attribution:       confidence-bucket / regime / cross analysis
  - threshold_adapter: baseline + addendum-aware threshold adaptation
  - walk_forward:      rolling train/test analysis with no-future-data invariant
  - edge_decay:        rolling-window edge-decay state machine
  - summary:           canonical learning-output dict assembly
"""

from hermes.learning.attribution import (
    AttributionConfig,
    BUCKET_BOUNDARIES,
    bucket_label_for_confidence,
    compute_attribution,
)
from hermes.learning.candidate_thresholds import (
    DEFAULT_THRESHOLD_SCHEMA_VERSION,
    REASON_CANDIDATE_REJECTED_OOS,
    REASON_EDGE_DECAY_ACTIVE,
    REASON_KILL_SWITCH_ACTIVE,
    REASON_NO_CANDIDATE,
    REASON_OUT_OF_BOUNDS,
    REASON_PROMOTED,
    ThresholdStore,
    promote_candidate,
    propose_candidate,
)
from hermes.learning.sample_size import (
    DEFAULT_MIN_TRADES_PER_BUCKET,
    DEFAULT_MIN_TRADES_PER_COMBINATION,
    DEFAULT_MIN_TRADES_PER_REGIME,
    DEFAULT_MIN_TRADES_TOTAL,
    REASON_INSUFFICIENT_SAMPLE_SIZE,
    SampleSizeConfig,
    adapt_thresholds_safely,
    check_sample_size,
)
from hermes.learning.oos_gate import (
    OOSValidationConfig,
    REASON_INSUFFICIENT_VALIDATION_TRADES,
    REASON_MAX_DRAWDOWN_INCREASED,
    REASON_PROFIT_FACTOR_DETERIORATED,
    REASON_STABILITY_TOO_LOW,
    REASON_TRADE_COUNT_REDUCED,
    REASON_VALIDATION_NOT_AFTER_DISCOVERY,
    REASON_WIN_RATE_BUT_RISK_WORSENS,
    evaluate_oos_promotion,
    split_trades_into_oos_windows,
)
from hermes.learning.performance_claim import (
    CLAIM_IMPROVED,
    CLAIM_INSUFFICIENT_EVIDENCE,
    CLAIM_PRESERVED,
    CLAIM_REJECTED,
    CLAIM_WORSENED,
    PerformanceClaimConfig,
    REASON_NO_VALIDATION_EVIDENCE,
    VALID_CLAIMS,
    evaluate_performance_claim,
)
from hermes.learning.edge_decay import EdgeDecayMonitor
from hermes.learning.memory import TradeMemory
from hermes.learning.summary import (
    REQUIRED_LEARNING_SUMMARY_KEYS,
    build_learning_summary,
)
from hermes.learning.threshold_adapter import (
    THRESHOLD_BOUNDS,
    ThresholdAdapter,
)
from hermes.learning.walk_forward import walk_forward_analysis

__all__ = [
    "AttributionConfig",
    "BUCKET_BOUNDARIES",
    "DEFAULT_MIN_TRADES_PER_BUCKET",
    "DEFAULT_MIN_TRADES_PER_COMBINATION",
    "DEFAULT_MIN_TRADES_PER_REGIME",
    "DEFAULT_MIN_TRADES_TOTAL",
    "DEFAULT_THRESHOLD_SCHEMA_VERSION",
    "EdgeDecayMonitor",
    "REASON_CANDIDATE_REJECTED_OOS",
    "REASON_EDGE_DECAY_ACTIVE",
    "REASON_INSUFFICIENT_SAMPLE_SIZE",
    "REASON_KILL_SWITCH_ACTIVE",
    "REASON_NO_CANDIDATE",
    "REASON_OUT_OF_BOUNDS",
    "REASON_PROMOTED",
    "REQUIRED_LEARNING_SUMMARY_KEYS",
    "SampleSizeConfig",
    "THRESHOLD_BOUNDS",
    "ThresholdAdapter",
    "ThresholdStore",
    "TradeMemory",
    "adapt_thresholds_safely",
    "bucket_label_for_confidence",
    "build_learning_summary",
    "check_sample_size",
    "compute_attribution",
    "promote_candidate",
    "propose_candidate",
    "walk_forward_analysis",
]
