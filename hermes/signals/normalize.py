"""Step 1 - Signal Normalization.

Validates and normalizes raw signal inputs deterministically.

Rules (per Prompt 1 Step 1):
  - amd_value, sequence_value, combined_value must be in [0, 1].
  - agreement = 1 - abs(sequence_value - amd_value).
  - No label logic (Step 2's job).
  - No implicit casting (booleans rejected; non-numerics rejected).
  - No hidden state. No randomness. No future data. No side effects.
  - Same input always produces same output.
"""

from hermes.utils.bounds import require_unit_interval

REQUIRED_SIGNAL_KEYS = ("sequence_value", "amd_value", "combined_value")


def validate_signal(raw):
    """Validate that `raw` is a dict containing the three required signal
    keys, each with a value in [0.0, 1.0].

    Raises ValueError with an actionable message on failure. Returns None on
    success (validation only).
    """
    if not isinstance(raw, dict):
        raise ValueError(
            "signal must be a dict; got {!s}".format(type(raw).__name__)
        )
    for key in REQUIRED_SIGNAL_KEYS:
        if key not in raw:
            raise ValueError("missing required signal key: {!r}".format(key))
        require_unit_interval(raw[key], key)
    return None


def compute_agreement(sequence_value, amd_value):
    """Return agreement = 1 - abs(sequence_value - amd_value), bounded [0,1].

    Both inputs must be real numbers in [0.0, 1.0]; otherwise ValueError.
    """
    s = require_unit_interval(sequence_value, "sequence_value")
    a = require_unit_interval(amd_value, "amd_value")
    return 1.0 - abs(s - a)


def normalize_signal(raw):
    """Validate `raw` and return a normalized signal dict.

    Output keys (exact set):
      - sequence_value: float in [0,1]
      - amd_value:      float in [0,1]
      - combined_value: float in [0,1]
      - agreement:      float in [0,1], = 1 - abs(sequence_value - amd_value)

    Does not mutate the input. Does not produce a label. Pure function.
    """
    validate_signal(raw)
    sequence_value = float(raw["sequence_value"])
    amd_value = float(raw["amd_value"])
    combined_value = float(raw["combined_value"])
    agreement = 1.0 - abs(sequence_value - amd_value)
    return {
        "sequence_value": sequence_value,
        "amd_value": amd_value,
        "combined_value": combined_value,
        "agreement": agreement,
    }
