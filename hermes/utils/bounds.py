"""Numeric bound helpers.

Pure stdlib. Deterministic. No implicit casting from non-numeric types.

`is_unit_interval` is the single source of truth for "x is a real number in
[0.0, 1.0]" used across the engine. Booleans are explicitly rejected because
`bool` is a subclass of `int` in Python and silent True->1.0 / False->0.0
coercion is a common source of bugs.
"""

_REAL_NUMERIC_TYPES = (int, float)


def is_real_number(x):
    """Return True iff x is a real number (int or float, not bool).

    Does NOT reject NaN. Use `is_numeric` if NaN must be rejected.
    Booleans are rejected because `bool` is a subclass of `int` in Python.
    """
    if isinstance(x, bool):
        return False
    return isinstance(x, _REAL_NUMERIC_TYPES)


def is_numeric(x):
    """Return True iff x is a real number (int or float, not bool) and not NaN.

    Rejects: bool, str, None, NaN, complex, and any non-numeric type.
    Accepts: int, float (including +/-inf, which are not NaN).
    """
    if not is_real_number(x):
        return False
    return x == x  # NaN check (NaN != NaN).


def clip01(x):
    """Clamp a real number to the closed unit interval [0.0, 1.0].

    Caller is responsible for ensuring x is numeric; this is a pure clamp.
    """
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def is_unit_interval(x):
    """Return True iff x is a real number in [0.0, 1.0].

    Rejects: bool, str, None, NaN, +/-inf, complex, and any non-numeric type.
    Accepts: int (0, 1) and float in the closed interval [0.0, 1.0].
    """
    if isinstance(x, bool):
        return False
    if not isinstance(x, _REAL_NUMERIC_TYPES):
        return False
    # NaN check (NaN != NaN). Also rules out +/-inf via the bounds check below.
    if x != x:
        return False
    return 0.0 <= float(x) <= 1.0


def require_unit_interval(x, name):
    """Validate that `x` is a real number in [0.0, 1.0] and return float(x).

    Raises ValueError with an actionable message on failure. The explicit
    float() cast is intentional; coercion from non-numeric types is rejected
    upstream by `is_unit_interval`.
    """
    if not is_unit_interval(x):
        raise ValueError(
            "value for {!s} must be a real number in [0.0, 1.0]; got {!r}".format(
                name, x
            )
        )
    return float(x)
