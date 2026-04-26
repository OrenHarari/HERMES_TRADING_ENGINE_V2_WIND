"""Safety hardening (Prompt 2 - Addendum).

Stand-alone safety validators and gates added by Prompt 2. This package is
ADDITIVE; it does not modify the Prompt 1 core engine or change any of the
existing public output dicts. Wiring of these validators into the top-level
make_decision() happens incrementally in subsequent Prompt 2 steps.

Step 0 (this file's first contribution): hermes/safety/data_contract.py.
"""

from hermes.safety.cost_model import (
    REASON_MISSING_COST_MODEL,
    VALID_FILL_MODELS,
    CostModel,
    apply_cost_model_to_trade,
    check_cost_model_for_live,
    validate_cost_model,
)
from hermes.safety.kill_switch import (
    KILL_SWITCH_REASONS,
    KS_REASON_ABNORMAL_SLIPPAGE,
    KS_REASON_ACCOUNT_EQUITY_UNAVAILABLE,
    KS_REASON_CORRUPTED_THRESHOLD_CONFIG,
    KS_REASON_CORRUPTED_TRADE_STATE,
    KS_REASON_COST_MODEL_MISSING,
    KS_REASON_DAILY_LOSS,
    KS_REASON_DUPLICATE_CANDLE,
    KS_REASON_EDGE_DECAY,
    KS_REASON_EXECUTION_ERROR,
    KS_REASON_INVALID_CANDLE,
    KS_REASON_MAX_CONSECUTIVE_LOSSES,
    KS_REASON_POSITION_STATE_MISMATCH,
    KS_REASON_STALE_DATA,
    KS_REASON_UNKNOWN_SYSTEM_MODE,
    KillSwitch,
    REASON_KILL_SWITCH_ACTIVE,
    ResetConfig,
    detect_abnormal_slippage,
    detect_corrupted_threshold_config,
    detect_position_state_mismatch,
)
from hermes.safety.paper_gate import (
    DETAIL_ACCOUNT_EQUITY_UNAVAILABLE,
    DETAIL_EDGE_DECAY_ACTIVE,
    DETAIL_INSUFFICIENT_PAPER_TRADES,
    DETAIL_INSUFFICIENT_REGIME_DIVERSITY,
    DETAIL_KILL_SWITCH_ACTIVE,
    DETAIL_MAX_DRAWDOWN_TOO_HIGH,
    DETAIL_MISSING_COST_MODEL,
    DETAIL_PROFIT_FACTOR_TOO_LOW,
    PaperGateConfig,
    REASON_LIVE_NOT_ENABLED,
    REASON_PAPER_VALIDATION_FAILED,
    check_live_trade_allowed,
    evaluate_paper_validation,
)
from hermes.safety.data_contract import (
    REASON_INVALID_DATA,
    REASON_INVALID_MODE,
    REASON_OK,
    REASON_STALE_DATA,
    SYSTEM_MODE_BACKTEST,
    SYSTEM_MODE_LIVE,
    SYSTEM_MODE_PAPER,
    VALID_SYSTEM_MODES,
    validate_candle_schema,
    validate_market_data,
    validate_system_mode,
)

__all__ = [
    "CostModel",
    "REASON_INVALID_DATA",
    "REASON_INVALID_MODE",
    "REASON_MISSING_COST_MODEL",
    "REASON_OK",
    "REASON_STALE_DATA",
    "SYSTEM_MODE_BACKTEST",
    "SYSTEM_MODE_LIVE",
    "SYSTEM_MODE_PAPER",
    "VALID_FILL_MODELS",
    "VALID_SYSTEM_MODES",
    "apply_cost_model_to_trade",
    "check_cost_model_for_live",
    "validate_candle_schema",
    "validate_cost_model",
    "validate_market_data",
    "validate_system_mode",
]
