"""Tests for Prompt 2 / Step 0 - Data Contract and Input Safety.

These tests cover the stand-alone safety validator. Wiring into the top-level
make_decision happens in later Prompt 2 steps; here we only verify that the
new validator produces the correct {"trade_allowed", "reason"} dicts for all
data-contract conditions.
"""

import unittest

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


def _candle(ts, o=100.0, h=101.0, l=99.0, c=100.5, v=None):
    d = {"timestamp": ts, "open": o, "high": h, "low": l, "close": c}
    if v is not None:
        d["volume"] = v
    return d


def _good_candles(n=5, start_ts=1000, step=60):
    return [_candle(start_ts + i * step) for i in range(n)]


# -------------------------------------------------------------------------
# System-mode validation
# -------------------------------------------------------------------------
class TestSystemModeValidation(unittest.TestCase):
    def test_valid_modes_accepted(self):
        for mode in VALID_SYSTEM_MODES:
            out = validate_system_mode(mode)
            self.assertTrue(out["trade_allowed"])
            self.assertEqual(out["reason"], REASON_OK)

    def test_three_explicit_modes_present(self):
        self.assertEqual(
            set(VALID_SYSTEM_MODES),
            {SYSTEM_MODE_BACKTEST, SYSTEM_MODE_PAPER, SYSTEM_MODE_LIVE},
        )

    def test_unknown_mode_rejected(self):
        out = validate_system_mode("trading_mode_9000")
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_MODE)

    def test_none_mode_treated_as_legacy_backtest(self):
        # Per addendum: "If legacy Prompt 1 code has no explicit system_mode
        # field, treat it as backtest_mode during migration only."
        out = validate_system_mode(None)
        self.assertTrue(out["trade_allowed"])

    def test_empty_string_mode_rejected(self):
        out = validate_system_mode("")
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_MODE)

    def test_non_string_mode_rejected(self):
        out = validate_system_mode(123)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_MODE)


# -------------------------------------------------------------------------
# Candle-schema validation (pure)
# -------------------------------------------------------------------------
class TestCandleSchemaValidation(unittest.TestCase):
    def test_valid_candles_pass(self):
        out = validate_candle_schema(_good_candles(), 4)
        self.assertTrue(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_OK)

    def test_empty_list_rejected(self):
        out = validate_candle_schema([], 0)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_current_index_out_of_range(self):
        out = validate_candle_schema(_good_candles(3), 5)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_missing_required_field(self):
        c = _good_candles()
        del c[2]["high"]
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_non_numeric_ohlc(self):
        c = _good_candles()
        c[1]["close"] = "100.5"
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_bool_ohlc_rejected(self):
        c = _good_candles()
        c[1]["close"] = True  # bool is subclass of int but must be rejected
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_nan_ohlc_rejected(self):
        c = _good_candles()
        c[1]["close"] = float("nan")
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_high_lt_low_rejected(self):
        c = _good_candles()
        c[2]["high"] = 50.0
        c[2]["low"] = 100.0
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_high_lt_open_rejected(self):
        c = _good_candles()
        c[2]["open"] = 200.0
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_high_lt_close_rejected(self):
        c = _good_candles()
        c[2]["close"] = 200.0
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_low_gt_open_rejected(self):
        c = _good_candles()
        c[2]["low"] = 200.0
        c[2]["open"] = 50.0
        # Now low > open AND high(101) < low(200): violates two rules; either
        # detection is fine. Must be rejected.
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_low_gt_close_rejected(self):
        c = _good_candles()
        # Make a subtle low > close violation while keeping high consistent.
        c[2] = {"timestamp": c[2]["timestamp"], "open": 100.0, "high": 105.0,
                "low": 102.0, "close": 101.0}
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_duplicate_timestamps_rejected(self):
        c = _good_candles()
        c[3]["timestamp"] = c[2]["timestamp"]
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_unsorted_timestamps_rejected(self):
        c = _good_candles()
        c[2]["timestamp"], c[3]["timestamp"] = c[3]["timestamp"], c[2]["timestamp"]
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_string_timestamps_accepted_when_sorted(self):
        # Per spec: "timestamp": int | str
        c = [_candle("2025-01-01T00:00:00Z"), _candle("2025-01-01T00:01:00Z")]
        # Adjust to make them unique strings:
        c[0]["timestamp"] = "2025-01-01T00:00:00Z"
        c[1]["timestamp"] = "2025-01-01T00:01:00Z"
        out = validate_candle_schema(c, 1)
        self.assertTrue(out["trade_allowed"])

    def test_volume_when_present_must_be_numeric(self):
        c = _good_candles()
        c[1]["volume"] = "1234"
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_volume_negative_rejected(self):
        c = _good_candles()
        c[1]["volume"] = -1.0
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_volume_when_present_zero_or_positive_ok(self):
        c = _good_candles()
        c[1]["volume"] = 0.0
        c[2]["volume"] = 1234.0
        out = validate_candle_schema(c, 4)
        self.assertTrue(out["trade_allowed"])

    def test_validation_uses_only_visible_window(self):
        # Future candles (index > current_index) may be invalid; visible
        # window must validate independently.
        c = _good_candles(5)
        c[4]["high"] = -999.0  # broken FUTURE candle (current_index=2)
        c[4]["low"] = -1000.0
        out = validate_candle_schema(c, 2)
        self.assertTrue(out["trade_allowed"])

    def test_non_dict_candle_rejected(self):
        c = _good_candles()
        c[1] = [100, 101, 99, 100.5]
        out = validate_candle_schema(c, 4)
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)


# -------------------------------------------------------------------------
# Future-data / lookahead rejection in validate_market_data
# -------------------------------------------------------------------------
class TestFutureDataRejection(unittest.TestCase):
    def test_future_candle_rejected_when_now_ts_provided(self):
        # Visible candle's timestamp > now_ts -> lookahead.
        c = _good_candles(5, start_ts=1000, step=60)
        # Last visible (index=4) timestamp = 1240. now_ts = 1100 -> lookahead.
        out = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_BACKTEST,
            now_ts=1100,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_present_candle_at_now_ts_accepted(self):
        c = _good_candles(5, start_ts=1000, step=60)
        out = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_BACKTEST,
            now_ts=1240,
        )
        self.assertTrue(out["trade_allowed"])

    def test_no_now_ts_means_no_lookahead_check(self):
        c = _good_candles(5)
        out = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_BACKTEST,
        )
        self.assertTrue(out["trade_allowed"])


# -------------------------------------------------------------------------
# Stale data behavior in paper/live; permissive in backtest
# -------------------------------------------------------------------------
class TestStaleDataRules(unittest.TestCase):
    def test_paper_mode_blocks_stale_data(self):
        c = _good_candles(5, start_ts=1000, step=60)
        # Last visible ts = 1240, now_ts = 5000, stale.
        out = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_PAPER,
            now_ts=5000, max_staleness_seconds=120,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_STALE_DATA)

    def test_live_mode_blocks_stale_data(self):
        c = _good_candles(5, start_ts=1000, step=60)
        out = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_LIVE,
            now_ts=5000, max_staleness_seconds=120,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_STALE_DATA)

    def test_backtest_mode_does_not_check_staleness(self):
        c = _good_candles(5, start_ts=1000, step=60)
        out = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_BACKTEST,
            now_ts=1_000_000_000, max_staleness_seconds=120,
        )
        self.assertTrue(out["trade_allowed"])

    def test_paper_mode_within_staleness_window_allowed(self):
        c = _good_candles(5, start_ts=1000, step=60)
        out = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_PAPER,
            now_ts=1300, max_staleness_seconds=120,
        )
        self.assertTrue(out["trade_allowed"])

    def test_paper_mode_at_exact_staleness_boundary_allowed(self):
        # Spec: stale if exceeds max; at exactly = max, allowed.
        c = _good_candles(5, start_ts=1000, step=60)
        out = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_PAPER,
            now_ts=1240 + 120, max_staleness_seconds=120,
        )
        self.assertTrue(out["trade_allowed"])

    def test_paper_mode_no_staleness_config_means_no_check(self):
        # If max_staleness_seconds isn't provided, paper mode does NOT
        # block on staleness (caller can opt in by providing the threshold).
        c = _good_candles(5, start_ts=1000, step=60)
        out = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_PAPER,
            now_ts=1_000_000_000,
        )
        self.assertTrue(out["trade_allowed"])

    def test_string_timestamps_skipped_for_staleness(self):
        # Staleness is a numeric comparison; if timestamps are strings, the
        # stale check is skipped (backtest semantics).
        c = [_candle("a"), _candle("b")]
        out = validate_market_data(
            c, current_index=1, system_mode=SYSTEM_MODE_PAPER,
            now_ts=999_999_999, max_staleness_seconds=120,
        )
        # Schema accepts; staleness skipped because timestamps aren't numeric.
        self.assertTrue(out["trade_allowed"])

    def test_staleness_check_is_deterministic(self):
        c = _good_candles(5, start_ts=1000, step=60)
        a = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_LIVE,
            now_ts=1_400, max_staleness_seconds=120,
        )
        b = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_LIVE,
            now_ts=1_400, max_staleness_seconds=120,
        )
        self.assertEqual(a, b)


# -------------------------------------------------------------------------
# Top-level validator integration: precedence + canonical output dict
# -------------------------------------------------------------------------
class TestValidateMarketDataPrecedence(unittest.TestCase):
    def test_invalid_mode_takes_precedence_over_data(self):
        c = _good_candles()
        out = validate_market_data(
            c, current_index=4, system_mode="bogus_mode",
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_MODE)

    def test_invalid_data_takes_precedence_over_staleness(self):
        c = _good_candles()
        c[2]["high"] = 0.0  # Broken OHLC.
        out = validate_market_data(
            c, current_index=4, system_mode=SYSTEM_MODE_PAPER,
            now_ts=10_000, max_staleness_seconds=10,
        )
        self.assertFalse(out["trade_allowed"])
        self.assertEqual(out["reason"], REASON_INVALID_DATA)

    def test_legacy_none_mode_uses_backtest_semantics(self):
        # No system_mode -> legacy/backtest: no staleness check.
        c = _good_candles(5, start_ts=1000, step=60)
        out = validate_market_data(
            c, current_index=4, system_mode=None,
            now_ts=1_000_000_000, max_staleness_seconds=120,
        )
        self.assertTrue(out["trade_allowed"])

    def test_output_shape_is_canonical(self):
        c = _good_candles()
        out = validate_market_data(c, 4, system_mode=SYSTEM_MODE_BACKTEST)
        self.assertEqual(set(out.keys()), {"trade_allowed", "reason"})
        self.assertIsInstance(out["trade_allowed"], bool)
        self.assertIsInstance(out["reason"], str)


if __name__ == "__main__":
    unittest.main()
