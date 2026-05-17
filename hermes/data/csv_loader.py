"""Phase 3A - CSV candle loader.

Pure stdlib. Loads OHLCV CSVs (with or without raw signal columns) and
returns a deterministic list of validated candle dicts ready to feed into
the offline backtest runner.

Two modes:

  MODE_WITH_SIGNALS - header has the 6 OHLCV columns plus
                      sequence_value, amd_value, combined_value.
                      Signal columns must each be in [0.0, 1.0].

  MODE_OHLCV_ONLY   - header has the 6 OHLCV columns only. The runner
                      will inject signals via baseline_signal() per bar.

Validation:
  - all required columns present (extra columns silently ignored)
  - every numeric cell parses; no NaN, no +/-inf
  - low <= open <= high, low <= close <= high
  - volume >= 0
  - timestamps strictly increasing (no duplicates, no out-of-order)
  - timestamps accepted as int seconds OR ISO-8601 strings (UTC normalized)
  - signal columns (mode A) inside [0, 1]

Any failure raises CsvLoadError with a precise reason string.
"""

import csv
import math
from datetime import datetime

REQUIRED_OHLCV_COLUMNS = (
    "timestamp", "open", "high", "low", "close", "volume",
)
SIGNAL_COLUMNS = ("sequence_value", "amd_value", "combined_value")

MODE_OHLCV_ONLY = "ohlcv_only"
MODE_WITH_SIGNALS = "with_signals"


class CsvLoadError(ValueError):
    """Raised on any CSV validation failure. Always carries a precise reason."""


# ---- helpers ------------------------------------------------------------

def _parse_float(value, field, row_num):
    if value is None:
        raise CsvLoadError(
            "row {0}: missing value for {1!r}".format(row_num, field)
        )
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise CsvLoadError(
            "row {0}: non-numeric value for {1!r}: {2!r}".format(
                row_num, field, value
            )
        )
    if math.isnan(v):
        raise CsvLoadError(
            "row {0}: NaN not allowed for {1!r}".format(row_num, field)
        )
    if math.isinf(v):
        raise CsvLoadError(
            "row {0}: +/-inf not allowed for {1!r}".format(row_num, field)
        )
    return v


def _parse_timestamp(value, row_num):
    """Return int seconds. Accepts int-like or ISO-8601 string."""
    if value is None or value == "":
        raise CsvLoadError(
            "row {0}: missing timestamp".format(row_num)
        )
    s = str(value).strip()
    # Try integer / float seconds first.
    try:
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            raise CsvLoadError(
                "row {0}: timestamp NaN/inf not allowed".format(row_num)
            )
        return int(f)
    except (TypeError, ValueError):
        pass
    # Fall through to ISO-8601.
    iso = s
    # datetime.fromisoformat in Python 3.11+ accepts 'Z'; normalize for older.
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        raise CsvLoadError(
            "row {0}: unparseable timestamp {1!r}".format(row_num, value)
        )
    # If naive, treat as UTC by convention.
    if dt.tzinfo is None:
        from datetime import timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _normalize_header(field_names):
    if field_names is None:
        return []
    return [
        (name.strip() if isinstance(name, str) else name)
        for name in field_names
    ]


def detect_mode(header):
    """Return MODE_WITH_SIGNALS or MODE_OHLCV_ONLY based on header columns.

    Raises CsvLoadError if required OHLCV columns are missing.
    """
    if header is None:
        raise CsvLoadError("CSV has no header row")
    cols = set(_normalize_header(header))
    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in cols]
    if missing:
        raise CsvLoadError(
            "CSV missing required column(s): {0}".format(", ".join(missing))
        )
    has_all_signals = all(c in cols for c in SIGNAL_COLUMNS)
    has_any_signal = any(c in cols for c in SIGNAL_COLUMNS)
    if has_all_signals:
        return MODE_WITH_SIGNALS
    if has_any_signal:
        # Partial signal columns: inconsistent header.
        present = [c for c in SIGNAL_COLUMNS if c in cols]
        absent = [c for c in SIGNAL_COLUMNS if c not in cols]
        raise CsvLoadError(
            "CSV has partial signal columns (present: {0}; missing: {1})".format(
                ", ".join(present), ", ".join(absent)
            )
        )
    return MODE_OHLCV_ONLY


# ---- main entry ---------------------------------------------------------

def load_candles_csv(path, *, expected_mode=None):
    """Load and validate a candle CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file (must exist; UTF-8 text).
    expected_mode : str | None
        If provided, must equal the auto-detected mode or CsvLoadError is
        raised.

    Returns
    -------
    dict
        {
            "mode": "with_signals" | "ohlcv_only",
            "candles": [...],     # list of validated candle dicts
            "row_count": int,
            "first_timestamp": int,
            "last_timestamp": int,
        }
    """
    if not isinstance(path, str) or not path:
        raise CsvLoadError("path must be a non-empty string")

    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            header = _normalize_header(reader.fieldnames)
            mode = detect_mode(header)
            if expected_mode is not None and expected_mode != mode:
                raise CsvLoadError(
                    "CSV mode mismatch: expected {0!r}, detected {1!r}".format(
                        expected_mode, mode
                    )
                )
            candles = []
            prev_ts = None
            for row_idx, raw in enumerate(reader, start=2):  # row 1 = header
                ts = _parse_timestamp(raw.get("timestamp"), row_idx)
                o = _parse_float(raw.get("open"), "open", row_idx)
                h = _parse_float(raw.get("high"), "high", row_idx)
                low = _parse_float(raw.get("low"), "low", row_idx)
                c = _parse_float(raw.get("close"), "close", row_idx)
                v = _parse_float(raw.get("volume"), "volume", row_idx)

                if v < 0.0:
                    raise CsvLoadError(
                        "row {0}: volume must be >= 0; got {1}".format(row_idx, v)
                    )
                if not (low <= o <= h):
                    raise CsvLoadError(
                        "row {0}: OHLC violation (low<=open<=high failed)".format(
                            row_idx
                        )
                    )
                if not (low <= c <= h):
                    raise CsvLoadError(
                        "row {0}: OHLC violation (low<=close<=high failed)".format(
                            row_idx
                        )
                    )
                if prev_ts is not None and ts <= prev_ts:
                    raise CsvLoadError(
                        "row {0}: timestamps must be strictly increasing "
                        "(prev={1}, current={2})".format(row_idx, prev_ts, ts)
                    )

                candle = {
                    "timestamp": ts,
                    "open": o,
                    "high": h,
                    "low": low,
                    "close": c,
                    "volume": v,
                }
                if mode == MODE_WITH_SIGNALS:
                    for sc in SIGNAL_COLUMNS:
                        sv = _parse_float(raw.get(sc), sc, row_idx)
                        if sv < 0.0 or sv > 1.0:
                            raise CsvLoadError(
                                "row {0}: {1} out of [0,1]: {2}".format(
                                    row_idx, sc, sv
                                )
                            )
                        candle[sc] = sv

                candles.append(candle)
                prev_ts = ts
    except FileNotFoundError:
        raise CsvLoadError("CSV file not found: {0!r}".format(path))
    except UnicodeDecodeError as exc:
        raise CsvLoadError("CSV is not valid UTF-8: {0}".format(exc))

    if not candles:
        raise CsvLoadError("CSV has zero data rows")

    return {
        "mode": mode,
        "candles": candles,
        "row_count": len(candles),
        "first_timestamp": candles[0]["timestamp"],
        "last_timestamp": candles[-1]["timestamp"],
    }


__all__ = [
    "CsvLoadError",
    "MODE_OHLCV_ONLY",
    "MODE_WITH_SIGNALS",
    "REQUIRED_OHLCV_COLUMNS",
    "SIGNAL_COLUMNS",
    "detect_mode",
    "load_candles_csv",
]
