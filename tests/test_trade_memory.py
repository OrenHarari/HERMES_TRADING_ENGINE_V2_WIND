"""Tests for Prompt 1 / Step 6 Part 1 - Trade Memory."""

import json
import os
import shutil
import tempfile
import unittest

from hermes.learning import TradeMemory
from hermes.lifecycle import build_completed_trade


def _entry(timestamp=1000, entry_price=100.0, **overrides):
    e = {
        "timestamp": timestamp,
        "date": "2025-01-15",
        "hour": 10,
        "entry_price": entry_price,
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


def _exit(pnl=5.0, exit_timestamp=2000, exit_price=105.0):
    return {"exit_timestamp": exit_timestamp, "exit_price": exit_price, "pnl": pnl}


def _record(timestamp=1000, entry_price=100.0, pnl=5.0):
    return build_completed_trade(_entry(timestamp, entry_price), _exit(pnl))


class TestTradeMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_mem_")
        self.path = os.path.join(self.tmp, "trade_memory.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_on_fresh_path(self):
        mem = TradeMemory(self.path)
        self.assertEqual(len(mem), 0)
        self.assertEqual(mem.all_trades(), [])

    def test_append_persists_to_disk(self):
        mem = TradeMemory(self.path)
        mem.append(_record(timestamp=1, entry_price=100.0))
        self.assertEqual(len(mem), 1)
        # File exists on disk and contains the record.
        self.assertTrue(os.path.exists(self.path))
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["timestamp"], 1)

    def test_reload_preserves_records(self):
        mem = TradeMemory(self.path)
        mem.append(_record(timestamp=1, entry_price=100.0))
        mem.append(_record(timestamp=2, entry_price=200.0))
        mem2 = TradeMemory(self.path)
        self.assertEqual(len(mem2), 2)

    def test_rejects_incomplete_record(self):
        mem = TradeMemory(self.path)
        bad = _record()
        del bad["regime"]
        with self.assertRaises(ValueError):
            mem.append(bad)

    def test_rejects_duplicate(self):
        mem = TradeMemory(self.path)
        mem.append(_record(timestamp=1, entry_price=100.0))
        with self.assertRaises(ValueError):
            mem.append(_record(timestamp=1, entry_price=100.0))

    def test_records_are_immutable_to_caller(self):
        mem = TradeMemory(self.path)
        rec = _record(timestamp=1)
        mem.append(rec)
        # Mutating the original dict must not change stored record.
        rec["pnl"] = 9999.0
        stored = mem.all_trades()[0]
        self.assertNotEqual(stored["pnl"], 9999.0)
        # And mutating returned dict must not change stored record either.
        stored["pnl"] = 8888.0
        again = mem.all_trades()[0]
        self.assertNotEqual(again["pnl"], 8888.0)

    def test_iter_yields_copies(self):
        mem = TradeMemory(self.path)
        mem.append(_record(timestamp=1))
        for r in mem:
            r["pnl"] = -1.0
        # No mutation should have leaked back.
        self.assertNotEqual(mem.all_trades()[0]["pnl"], -1.0)

    def test_load_rejects_non_list_file(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"not": "a list"}, fh)
        with self.assertRaises(ValueError):
            TradeMemory(self.path)

    def test_load_rejects_corrupted_record(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump([{"timestamp": 1}], fh)  # missing required keys
        with self.assertRaises(ValueError):
            TradeMemory(self.path)

    def test_dedup_distinguishes_by_entry_price(self):
        mem = TradeMemory(self.path)
        mem.append(_record(timestamp=1, entry_price=100.0))
        # Same timestamp, different entry_price -> allowed.
        mem.append(_record(timestamp=1, entry_price=101.0))
        self.assertEqual(len(mem), 2)


if __name__ == "__main__":
    unittest.main()
