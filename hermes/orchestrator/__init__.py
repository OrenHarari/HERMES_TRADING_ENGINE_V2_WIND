"""Orchestrator layer (Prompt 1, Step 2).

Composes the canonical signal-layer output dict from normalized values.
Pure function. No side effects. No mutable external state.
"""

from hermes.orchestrator.orchestrator import (
    LABEL_NEUTRAL,
    LABEL_STRONG,
    LABEL_VALUES,
    LABEL_WEAK,
    REQUIRED_OUTPUT_KEYS,
    build_signal_output,
    derive_label,
)

__all__ = [
    "LABEL_NEUTRAL",
    "LABEL_STRONG",
    "LABEL_VALUES",
    "LABEL_WEAK",
    "REQUIRED_OUTPUT_KEYS",
    "build_signal_output",
    "derive_label",
]
