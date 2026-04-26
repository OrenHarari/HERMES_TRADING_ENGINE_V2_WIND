"""Step 2 - Orchestrator Consistency.

Builds the canonical signal-layer output dict. The label is derived
deterministically and ONLY from normalized signal values - no external state,
no randomness, no hidden inputs.

Output keys (exact set):
  - sequence_value
  - amd_value
  - combined_value
  - agreement
  - label

Label thresholds are applied to `combined_value`:
  - combined_value <  LABEL_WEAK_UPPER     -> "weak"
  - LABEL_WEAK_UPPER <= combined_value < LABEL_STRONG_LOWER -> "neutral"
  - combined_value >= LABEL_STRONG_LOWER   -> "strong"

Boundaries are inclusive on the upper buckets to make the function total and
deterministic at the exact threshold values.
"""

from hermes.signals import normalize_signal

LABEL_WEAK = "weak"
LABEL_NEUTRAL = "neutral"
LABEL_STRONG = "strong"
LABEL_VALUES = (LABEL_WEAK, LABEL_NEUTRAL, LABEL_STRONG)

# Threshold boundaries for label derivation (deterministic, value-only).
LABEL_WEAK_UPPER = 0.4   # combined_value < 0.4  -> weak
LABEL_STRONG_LOWER = 0.6  # combined_value >= 0.6 -> strong

REQUIRED_OUTPUT_KEYS = (
    "sequence_value",
    "amd_value",
    "combined_value",
    "agreement",
    "label",
)


def derive_label(combined_value):
    """Derive a label from `combined_value` only.

    Pure, total function over [0.0, 1.0]. No external state. No randomness.
    Caller is responsible for ensuring `combined_value` is already validated
    by the signals layer; this function does not re-validate.
    """
    if combined_value < LABEL_WEAK_UPPER:
        return LABEL_WEAK
    if combined_value < LABEL_STRONG_LOWER:
        return LABEL_NEUTRAL
    return LABEL_STRONG


def build_signal_output(raw):
    """Validate `raw`, normalize it, and return the canonical orchestrator dict.

    The dict has exactly the keys in REQUIRED_OUTPUT_KEYS, all values in
    [0.0, 1.0] for the numeric ones, and `label` in LABEL_VALUES.

    Pure function: same input always produces same output. The input dict is
    not mutated.
    """
    normalized = normalize_signal(raw)
    label = derive_label(normalized["combined_value"])
    return {
        "sequence_value": normalized["sequence_value"],
        "amd_value": normalized["amd_value"],
        "combined_value": normalized["combined_value"],
        "agreement": normalized["agreement"],
        "label": label,
    }
