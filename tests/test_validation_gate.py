"""Tests for Prompt 1 / Step 5 Part 4 - Backtest Validation Gate."""

import unittest

from hermes.decision.validation import (
    assert_deterministic_replay,
    assert_no_future_data,
    validate_backtest,
)


def _good_decision(candles, current_index):
    # Pure function over past+present only.
    window = candles[: current_index + 1]
    return {"sum_close": sum(c["close"] for c in window), "n": len(window)}


def _leaky_decision(candles, current_index):
    # Reads ALL candles -> leaks future data.
    return {"sum_close": sum(c["close"] for c in candles), "n": len(candles)}


def _stateful_decision_factory():
    state = {"calls": 0}

    def fn(candles, current_index):
        state["calls"] += 1
        return {"call": state["calls"], "idx": current_index}

    return fn


def _candles(n=10):
    return [{"open": float(i), "high": float(i) + 0.5, "low": float(i) - 0.5, "close": float(i)} for i in range(1, n + 1)]


def _garbage(idx):
    return {"open": 1e9, "high": 1e9, "low": 1e9, "close": 1e9}


class TestAssertDeterministicReplay(unittest.TestCase):
    def test_passes_for_deterministic_function(self):
        out = assert_deterministic_replay(_good_decision, _candles(), [0, 3, 7, 9])
        self.assertTrue(out["validation_passed"])
        self.assertEqual(out["reason"], "")

    def test_fails_for_stateful_function(self):
        fn = _stateful_decision_factory()
        out = assert_deterministic_replay(fn, _candles(), [0, 1, 2])
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], "non_deterministic_replay")


class TestAssertNoFutureData(unittest.TestCase):
    def test_passes_for_pure_decision_function(self):
        out = assert_no_future_data(_good_decision, _candles(), 4, _garbage)
        self.assertTrue(out["validation_passed"])

    def test_fails_for_leaky_decision_function(self):
        out = assert_no_future_data(_leaky_decision, _candles(), 4, _garbage)
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], "future_data_leakage")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            assert_no_future_data(_good_decision, "not list", 0, _garbage)
        with self.assertRaises(ValueError):
            assert_no_future_data(_good_decision, _candles(), "x", _garbage)
        with self.assertRaises(ValueError):
            assert_no_future_data(_good_decision, _candles(), 999, _garbage)


class TestValidateBacktest(unittest.TestCase):
    def test_passes_for_pure_function(self):
        out = validate_backtest(
            _good_decision,
            candles_factory=lambda: _candles(),
            indices=[0, 3, 7],
            garbage_factory=_garbage,
        )
        self.assertTrue(out["validation_passed"])

    def test_fails_for_leaky_function(self):
        out = validate_backtest(
            _leaky_decision,
            candles_factory=lambda: _candles(),
            indices=[0, 3, 7],
            garbage_factory=_garbage,
        )
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], "future_data_leakage")

    def test_fails_for_non_deterministic_function(self):
        fn = _stateful_decision_factory()
        out = validate_backtest(
            fn,
            candles_factory=lambda: _candles(),
            indices=[0, 1, 2],
            garbage_factory=_garbage,
        )
        self.assertFalse(out["validation_passed"])
        self.assertEqual(out["reason"], "non_deterministic_replay")


if __name__ == "__main__":
    unittest.main()
