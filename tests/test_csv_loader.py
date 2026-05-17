"""Tests for hermes.data.csv_loader (Phase 3A)."""

import os
import tempfile
import unittest

from hermes.data.csv_loader import (
    CsvLoadError,
    MODE_OHLCV_ONLY,
    MODE_WITH_SIGNALS,
    detect_mode,
    load_candles_csv,
)

FIXTURES = os.path.join(
    os.path.dirname(__file__), "fixtures", "phase3a"
)


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


class TestDetectMode(unittest.TestCase):
    def test_with_signals(self):
        h = ["timestamp", "open", "high", "low", "close", "volume",
             "sequence_value", "amd_value", "combined_value"]
        self.assertEqual(detect_mode(h), MODE_WITH_SIGNALS)

    def test_ohlcv_only(self):
        h = ["timestamp", "open", "high", "low", "close", "volume"]
        self.assertEqual(detect_mode(h), MODE_OHLCV_ONLY)

    def test_missing_required_column(self):
        h = ["timestamp", "open", "high", "low", "close"]  # no volume
        with self.assertRaises(CsvLoadError):
            detect_mode(h)

    def test_partial_signals_rejected(self):
        h = ["timestamp", "open", "high", "low", "close", "volume",
             "sequence_value"]  # missing amd_value, combined_value
        with self.assertRaises(CsvLoadError):
            detect_mode(h)

    def test_extra_columns_ignored(self):
        h = ["timestamp", "open", "high", "low", "close", "volume", "foo"]
        self.assertEqual(detect_mode(h), MODE_OHLCV_ONLY)


class TestLoadCandlesCsv(unittest.TestCase):
    def test_happy_mode_a_with_signals(self):
        out = load_candles_csv(
            os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv")
        )
        self.assertEqual(out["mode"], MODE_WITH_SIGNALS)
        self.assertEqual(out["row_count"], 30)
        self.assertEqual(out["first_timestamp"], 1700000000)
        self.assertEqual(out["last_timestamp"], 1700104400)
        c = out["candles"][0]
        for k in ("timestamp", "open", "high", "low", "close", "volume",
                  "sequence_value", "amd_value", "combined_value"):
            self.assertIn(k, c)
        self.assertIsInstance(c["timestamp"], int)
        self.assertIsInstance(c["open"], float)

    def test_happy_mode_b_ohlcv_only(self):
        out = load_candles_csv(
            os.path.join(FIXTURES, "AMD_1h_mode_b_ohlcv.csv")
        )
        self.assertEqual(out["mode"], MODE_OHLCV_ONLY)
        self.assertEqual(out["row_count"], 30)
        c = out["candles"][0]
        self.assertNotIn("sequence_value", c)

    def test_expected_mode_match_passes(self):
        out = load_candles_csv(
            os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv"),
            expected_mode=MODE_WITH_SIGNALS,
        )
        self.assertEqual(out["mode"], MODE_WITH_SIGNALS)

    def test_expected_mode_mismatch_rejected(self):
        with self.assertRaises(CsvLoadError):
            load_candles_csv(
                os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv"),
                expected_mode=MODE_OHLCV_ONLY,
            )

    def test_missing_required_column_rejected(self):
        with self.assertRaises(CsvLoadError):
            load_candles_csv(
                os.path.join(FIXTURES, "invalid_missing_columns.csv")
            )

    def test_unsorted_timestamps_rejected(self):
        with self.assertRaises(CsvLoadError) as ctx:
            load_candles_csv(os.path.join(FIXTURES, "invalid_unsorted.csv"))
        self.assertIn("strictly increasing", str(ctx.exception))

    def test_duplicate_timestamps_rejected(self):
        with self.assertRaises(CsvLoadError) as ctx:
            load_candles_csv(
                os.path.join(FIXTURES, "invalid_duplicate_ts.csv")
            )
        self.assertIn("strictly increasing", str(ctx.exception))

    def test_ohlc_violation_rejected(self):
        with self.assertRaises(CsvLoadError) as ctx:
            load_candles_csv(
                os.path.join(FIXTURES, "invalid_ohlc_violation.csv")
            )
        self.assertIn("OHLC", str(ctx.exception))

    def test_nonexistent_file_rejected(self):
        with self.assertRaises(CsvLoadError):
            load_candles_csv(os.path.join(FIXTURES, "does_not_exist.csv"))

    def test_deterministic(self):
        path = os.path.join(FIXTURES, "AMD_1h_mode_a_signals.csv")
        a = load_candles_csv(path)
        b = load_candles_csv(path)
        self.assertEqual(a, b)


class TestNumericValidationViaTempFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_csv_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def test_non_numeric_cell_rejected(self):
        p = self._path("bad.csv")
        _write(p,
               "timestamp,open,high,low,close,volume\n"
               "1700000000,abc,100.5,99.8,100.2,1500\n")
        with self.assertRaises(CsvLoadError) as ctx:
            load_candles_csv(p)
        self.assertIn("non-numeric", str(ctx.exception))

    def test_negative_volume_rejected(self):
        p = self._path("neg_vol.csv")
        _write(p,
               "timestamp,open,high,low,close,volume\n"
               "1700000000,100.0,100.5,99.8,100.2,-10\n")
        with self.assertRaises(CsvLoadError) as ctx:
            load_candles_csv(p)
        self.assertIn("volume", str(ctx.exception))

    def test_signal_out_of_range_rejected(self):
        p = self._path("bad_sig.csv")
        _write(p,
               "timestamp,open,high,low,close,volume,"
               "sequence_value,amd_value,combined_value\n"
               "1700000000,100.0,100.5,99.8,100.2,1500,1.5,0.5,0.5\n")
        with self.assertRaises(CsvLoadError) as ctx:
            load_candles_csv(p)
        self.assertIn("[0,1]", str(ctx.exception))

    def test_empty_file_rejected(self):
        p = self._path("empty.csv")
        _write(p, "timestamp,open,high,low,close,volume\n")
        with self.assertRaises(CsvLoadError) as ctx:
            load_candles_csv(p)
        self.assertIn("zero data rows", str(ctx.exception))

    def test_iso_timestamp_parsed(self):
        p = self._path("iso.csv")
        _write(p,
               "timestamp,open,high,low,close,volume\n"
               "2024-01-01T00:00:00+00:00,100.0,100.5,99.8,100.2,1500\n"
               "2024-01-01T01:00:00+00:00,100.2,100.8,100.1,100.7,1400\n")
        out = load_candles_csv(p)
        self.assertEqual(out["row_count"], 2)
        self.assertEqual(
            out["candles"][1]["timestamp"] - out["candles"][0]["timestamp"],
            3600,
        )

    def test_iso_timestamp_with_z_suffix(self):
        p = self._path("isoz.csv")
        _write(p,
               "timestamp,open,high,low,close,volume\n"
               "2024-01-01T00:00:00Z,100.0,100.5,99.8,100.2,1500\n"
               "2024-01-01T01:00:00Z,100.2,100.8,100.1,100.7,1400\n")
        out = load_candles_csv(p)
        self.assertEqual(out["row_count"], 2)


if __name__ == "__main__":
    unittest.main()
