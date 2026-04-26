"""Step 6 / Part 1 - Trade Memory.

Append-only JSON store for completed trades. Validates every appended record
through the lifecycle.completed_trade builder/validator so that incomplete
trades CANNOT enter memory.

Records are immutable once written. Dedup by (timestamp, entry_price).

Storage: a single JSON file (default `data/trade_memory.json`). Pure stdlib.
"""

import os

from hermes.lifecycle.completed_trade import (
    REQUIRED_TRADE_KEYS,
    is_complete_trade_record,
)
from hermes.utils.json_io import read_json, write_json_atomic

DEFAULT_MEMORY_PATH = os.path.join("data", "trade_memory.json")


def _dedup_key(record):
    return (record["timestamp"], record["entry_price"])


class TradeMemory(object):
    """Append-only completed-trade store backed by a JSON file."""

    def __init__(self, path=None):
        self._path = path if path is not None else DEFAULT_MEMORY_PATH
        self._records = self._load()
        self._dedup = {_dedup_key(r) for r in self._records}

    @property
    def path(self):
        return self._path

    def _load(self):
        data = read_json(self._path, default=[])
        if not isinstance(data, list):
            raise ValueError(
                "trade memory at {!s} is not a JSON list".format(self._path)
            )
        # Reject any record that doesn't pass the lifecycle validator -
        # protects against silent corruption.
        for i, r in enumerate(data):
            if not is_complete_trade_record(r):
                raise ValueError(
                    "trade memory record {!s} is incomplete or invalid".format(i)
                )
        return list(data)

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        # Yield copies so callers cannot mutate stored records.
        for r in self._records:
            yield dict(r)

    def all_trades(self):
        """Return a list of dict copies (immutability guarantee)."""
        return [dict(r) for r in self._records]

    def append(self, record):
        """Append a validated completed-trade record. Raises ValueError if:
          - record is incomplete / invalid
          - record duplicates an existing (timestamp, entry_price) key
        """
        if not is_complete_trade_record(record):
            raise ValueError("cannot append incomplete or invalid trade record")
        # Force-validate via REQUIRED_TRADE_KEYS - extra defensive layer.
        for k in REQUIRED_TRADE_KEYS:
            if k not in record:
                raise ValueError(
                    "trade record missing required key: {!r}".format(k)
                )
        key = _dedup_key(record)
        if key in self._dedup:
            raise ValueError(
                "duplicate trade record for (timestamp={!r}, entry_price={!r})".format(
                    record["timestamp"], record["entry_price"]
                )
            )
        # Take a defensive copy to ensure immutability.
        stored = dict(record)
        self._records.append(stored)
        self._dedup.add(key)
        write_json_atomic(self._path, self._records)
