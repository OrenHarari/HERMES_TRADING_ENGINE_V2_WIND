"""Phase 3A - Data provenance.

Every report MUST include a `data_provenance` block. This module is the
single source of truth for that block's shape and warnings.

Source enum (canonical, do not rename):

  DATA_SOURCE_USER_PROVIDED       = "user_provided"
  DATA_SOURCE_TEST_FIXTURE        = "test_fixture_synthetic"

Anything else is rejected by `build_data_provenance` -- there is no
"unknown" source, because anonymous data without provenance is exactly
what the rule is meant to prevent.

Warnings (verbatim text the user spec mandates):
"""

DATA_SOURCE_USER_PROVIDED = "user_provided"
DATA_SOURCE_TEST_FIXTURE = "test_fixture_synthetic"

VALID_SOURCES = (DATA_SOURCE_USER_PROVIDED, DATA_SOURCE_TEST_FIXTURE)

WARNING_USER_PROVIDED = (
    "User-provided local CSV - source not independently verified by HERMES."
)
WARNING_TEST_FIXTURE = (
    "Synthetic test fixture - not real market data."
)

DATA_PROVENANCE_KEYS = (
    "file_path",
    "is_synthetic",
    "source",
    "symbol",
    "timeframe",
    "row_count",
    "start_timestamp",
    "end_timestamp",
    "warning",
)


def build_data_provenance(
    *,
    file_path,
    symbol,
    timeframe,
    row_count,
    start_timestamp,
    end_timestamp,
    source=DATA_SOURCE_USER_PROVIDED,
):
    """Build a canonical data_provenance dict.

    Parameters
    ----------
    file_path : str
        Absolute or relative path of the CSV that was loaded.
    symbol, timeframe : str
        Labels echoed from the runner config.
    row_count : int
        Number of candle rows (must be >= 1).
    start_timestamp, end_timestamp : int
        First and last candle timestamps (seconds).
    source : str
        Must be one of:
          - "user_provided"          (real local CSV from the operator)
          - "test_fixture_synthetic" (small hand-crafted CSV under
                                      tests/fixtures)
    Returns
    -------
    dict
        With exactly the keys in `DATA_PROVENANCE_KEYS`.

    Raises
    ------
    ValueError on invalid source / missing arguments / non-numeric
    timestamps / non-positive row_count.
    """
    if source not in VALID_SOURCES:
        raise ValueError(
            "data_provenance.source must be one of {0!r}; got {1!r}".format(
                VALID_SOURCES, source
            )
        )
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol must be a non-empty string")
    if not isinstance(timeframe, str) or not timeframe:
        raise ValueError("timeframe must be a non-empty string")
    if not isinstance(row_count, int) or row_count < 1:
        raise ValueError("row_count must be a positive int")
    if not isinstance(start_timestamp, int):
        raise ValueError("start_timestamp must be an int")
    if not isinstance(end_timestamp, int):
        raise ValueError("end_timestamp must be an int")
    if end_timestamp < start_timestamp:
        raise ValueError("end_timestamp must be >= start_timestamp")

    if source == DATA_SOURCE_TEST_FIXTURE:
        is_synthetic = True
        warning = WARNING_TEST_FIXTURE
    else:
        is_synthetic = False
        warning = WARNING_USER_PROVIDED

    return {
        "file_path": file_path,
        "is_synthetic": is_synthetic,
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
        "row_count": row_count,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "warning": warning,
    }


__all__ = [
    "DATA_PROVENANCE_KEYS",
    "DATA_SOURCE_TEST_FIXTURE",
    "DATA_SOURCE_USER_PROVIDED",
    "VALID_SOURCES",
    "WARNING_TEST_FIXTURE",
    "WARNING_USER_PROVIDED",
    "build_data_provenance",
]
