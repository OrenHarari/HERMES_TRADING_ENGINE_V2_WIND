"""Lifecycle layer.

Phase 1 (Prompt 1): MINIMAL completed-trade builder consumed by
`learning/memory.py`. Validates that a record has the required keys and
derives a canonical 'outcome' string. This guarantees that incomplete trades
cannot enter trade memory.

Phase 2 (Prompt 2 Step 5B): this folder will be expanded into a full
trade-lifecycle FSM (entry -> open -> exit) with cost model integration. The
schema produced here is a strict subset of the Phase-2 schema so additions
are purely additive.
"""

from hermes.lifecycle.completed_trade import (
    OUTCOME_BREAKEVEN,
    OUTCOME_LOSS,
    OUTCOME_VALUES,
    OUTCOME_WIN,
    REQUIRED_TRADE_KEYS,
    build_completed_trade,
    is_complete_trade_record,
)
from hermes.lifecycle.exit_rules import (
    EXIT_REASONS,
    EXIT_REASON_END_OF_BACKTEST,
    EXIT_REASON_MAX_HOLDING,
    EXIT_REASON_RISK_GUARDRAIL,
    EXIT_REASON_SIGNAL_DECAY,
    EXIT_REASON_STOP_LOSS,
    EXIT_REASON_TAKE_PROFIT,
    ExitRulesConfig,
    decide_exit,
)
from hermes.lifecycle.lifecycle import (
    REQUIRED_EXIT_FIELDS,
    REQUIRED_OPEN_POSITION_META_KEYS,
    OpenPosition,
    complete_trade,
)

__all__ = [
    "EXIT_REASONS",
    "EXIT_REASON_END_OF_BACKTEST",
    "EXIT_REASON_MAX_HOLDING",
    "EXIT_REASON_RISK_GUARDRAIL",
    "EXIT_REASON_SIGNAL_DECAY",
    "EXIT_REASON_STOP_LOSS",
    "EXIT_REASON_TAKE_PROFIT",
    "ExitRulesConfig",
    "OUTCOME_BREAKEVEN",
    "OUTCOME_LOSS",
    "OUTCOME_VALUES",
    "OUTCOME_WIN",
    "OpenPosition",
    "REQUIRED_EXIT_FIELDS",
    "REQUIRED_OPEN_POSITION_META_KEYS",
    "REQUIRED_TRADE_KEYS",
    "build_completed_trade",
    "complete_trade",
    "decide_exit",
    "is_complete_trade_record",
]
