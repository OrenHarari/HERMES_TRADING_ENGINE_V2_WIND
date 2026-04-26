"""Tests for the lifecycle/completed_trade builder (Phase 1 minimal).

These also cover the requirement that incomplete trades cannot enter memory.
"""

import unittest

from hermes.lifecycle import (
    OUTCOME_BREAKEVEN,
    OUTCOME_LOSS,
    OUTCOME_WIN,
    REQUIRED_TRADE_KEYS,
    build_completed_trade,
    is_complete_trade_record,
)


def _good_entry(**overrides):
    e = {
        "timestamp": 1000,
        "date": "2025-01-15",
        "hour": 10,
        "entry_price": 100.0,
        "sequence_value": 0.7,
        "amd_value": 0.7,
        "combined_value": 0.7,
        "agreement": 1.0,
        "confidence": 0.75,
        "regime": "trend_up",
        "momentum_score": 0.7,
        "volatility_score": 0.4,
    }
    e.update(overrides)
    return e


def _good_exit(pnl=5.0, **overrides):
    x = {
        "exit_timestamp": 2000,
        "exit_price": 105.0,
        "pnl": pnl,
    }
    x.update(overrides)
    return x


class TestBuildCompletedTrade(unittest.TestCase):
    def test_required_keys_present(self):
        rec = build_completed_trade(_good_entry(), _good_exit(5.0))
        for k in REQUIRED_TRADE_KEYS:
            self.assertIn(k, rec)

    def test_outcome_win(self):
        rec = build_completed_trade(_good_entry(), _good_exit(10.0))
        self.assertEqual(rec["outcome"], OUTCOME_WIN)

    def test_outcome_loss(self):
        rec = build_completed_trade(_good_entry(), _good_exit(-3.0))
        self.assertEqual(rec["outcome"], OUTCOME_LOSS)

    def test_outcome_breakeven(self):
        rec = build_completed_trade(_good_entry(), _good_exit(0.0))
        self.assertEqual(rec["outcome"], OUTCOME_BREAKEVEN)

    def test_optional_net_pnl_passthrough(self):
        rec = build_completed_trade(
            _good_entry(), _good_exit(5.0, net_pnl=3.0)
        )
        self.assertEqual(rec["net_pnl"], 3.0)

    def test_optional_notes_passthrough(self):
        rec = build_completed_trade(
            _good_entry(), _good_exit(5.0, notes="manual")
        )
        self.assertEqual(rec["notes"], "manual")

    def test_rejects_missing_entry_field(self):
        e = _good_entry()
        del e["regime"]
        with self.assertRaises(ValueError):
            build_completed_trade(e, _good_exit(1.0))

    def test_rejects_invalid_regime(self):
        with self.assertRaises(ValueError):
            build_completed_trade(_good_entry(regime="bogus"), _good_exit(1.0))

    def test_rejects_out_of_range_signal(self):
        with self.assertRaises(ValueError):
            build_completed_trade(_good_entry(confidence=1.5), _good_exit(1.0))

    def test_rejects_exit_before_entry(self):
        with self.assertRaises(ValueError):
            build_completed_trade(
                _good_entry(timestamp=2000),
                _good_exit(1.0, exit_timestamp=1500),
            )

    def test_allows_exit_at_same_timestamp_as_entry(self):
        # Equal timestamps are accepted (atomic-fill semantics).
        rec = build_completed_trade(
            _good_entry(timestamp=2000),
            _good_exit(1.0, exit_timestamp=2000),
        )
        self.assertEqual(rec["timestamp"], 2000)
        self.assertEqual(rec["exit_timestamp"], 2000)

    def test_rejects_non_positive_prices(self):
        with self.assertRaises(ValueError):
            build_completed_trade(_good_entry(entry_price=0.0), _good_exit(1.0))
        with self.assertRaises(ValueError):
            build_completed_trade(_good_entry(), _good_exit(1.0, exit_price=-1.0))

    def test_rejects_invalid_hour(self):
        with self.assertRaises(ValueError):
            build_completed_trade(_good_entry(hour=24), _good_exit(1.0))
        with self.assertRaises(ValueError):
            build_completed_trade(_good_entry(hour=-1), _good_exit(1.0))

    def test_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            build_completed_trade("not a dict", _good_exit(1.0))
        with self.assertRaises(ValueError):
            build_completed_trade(_good_entry(), "not a dict")


class TestIsCompleteTradeRecord(unittest.TestCase):
    def test_returns_true_for_valid(self):
        rec = build_completed_trade(_good_entry(), _good_exit(1.0))
        self.assertTrue(is_complete_trade_record(rec))

    def test_returns_false_for_missing_key(self):
        rec = build_completed_trade(_good_entry(), _good_exit(1.0))
        del rec["pnl"]
        self.assertFalse(is_complete_trade_record(rec))

    def test_returns_false_for_invalid_value(self):
        rec = build_completed_trade(_good_entry(), _good_exit(1.0))
        rec["confidence"] = 1.5
        self.assertFalse(is_complete_trade_record(rec))

    def test_returns_false_for_non_dict(self):
        self.assertFalse(is_complete_trade_record(None))
        self.assertFalse(is_complete_trade_record([]))
        self.assertFalse(is_complete_trade_record("trade"))


if __name__ == "__main__":
    unittest.main()
