"""Signal normalization layer (Prompt 1, Step 1).

Validates and normalizes raw signal inputs. Does NOT derive labels - that is
the orchestrator's responsibility (Step 2).
"""

from hermes.signals.normalize import (
    REQUIRED_SIGNAL_KEYS,
    compute_agreement,
    normalize_signal,
    validate_signal,
)

__all__ = [
    "REQUIRED_SIGNAL_KEYS",
    "compute_agreement",
    "normalize_signal",
    "validate_signal",
]
