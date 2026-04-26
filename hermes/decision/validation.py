"""Step 5 / Part 4 - Backtest Validation Gate.

Checks for:
  - deterministic replay (same input produces same output, twice)
  - no future-data usage (mutating future candles must not change output)

Returns: {"validation_passed": bool, "reason": str}.

The decision function under test must have signature:
    fn(candles, current_index) -> hashable / dict
where the result is comparable for equality.
"""


def assert_deterministic_replay(decision_fn, candles, indices):
    """Run `decision_fn` twice on the same inputs and compare results.

    Returns {"validation_passed": bool, "reason": str}.
    """
    first = []
    for i in indices:
        first.append(decision_fn(candles, i))
    second = []
    for i in indices:
        second.append(decision_fn(candles, i))
    if first != second:
        return {
            "validation_passed": False,
            "reason": "non_deterministic_replay",
        }
    return {"validation_passed": True, "reason": ""}


def assert_no_future_data(decision_fn, candles, current_index, garbage_factory):
    """Verify that mutating candles[current_index+1:] does not change output.

    `garbage_factory(idx)` produces a replacement candle for index `idx`.
    The original candles list is mutated in place by this assertion (callers
    must pass a fresh list).
    """
    if not isinstance(candles, list):
        raise ValueError("candles must be a list")
    if not isinstance(current_index, int):
        raise ValueError("current_index must be int")
    if current_index >= len(candles):
        raise ValueError("current_index out of range")

    before = decision_fn(candles, current_index)
    for j in range(current_index + 1, len(candles)):
        candles[j] = garbage_factory(j)
    after = decision_fn(candles, current_index)
    if before != after:
        return {
            "validation_passed": False,
            "reason": "future_data_leakage",
        }
    return {"validation_passed": True, "reason": ""}


def validate_backtest(decision_fn, candles_factory, indices, garbage_factory):
    """Run both replay and no-future-data checks.

    `candles_factory()` must produce a fresh equal candle list per call so the
    no-future-data probe can mutate it without affecting other probes.

    Returns {"validation_passed": bool, "reason": str}.
    """
    candles_a = candles_factory()
    res = assert_deterministic_replay(decision_fn, candles_a, indices)
    if not res["validation_passed"]:
        return res
    for idx in indices:
        candles_b = candles_factory()
        res = assert_no_future_data(decision_fn, candles_b, idx, garbage_factory)
        if not res["validation_passed"]:
            return res
    return {"validation_passed": True, "reason": ""}
