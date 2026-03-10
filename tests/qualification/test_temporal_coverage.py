"""."""

from datetime import UTC, datetime, timedelta, timezone

import polars as pl
import pytest

from omega_prime.qualification.temporal_coverage import (
    TEMPORAL_COVERAGE,
    _ensure_utc,
    _parse_datetime,
    temporal_coverage,
)

from .conftest import qualification_assert


@pytest.fixture()
def temporal_coverage_df() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "total_nanos": [0, int(10 * 1e9)],  # 0s and 10s in nanoseconds since epoch
        }
    ).lazy()


def test_pass(temporal_coverage_df: pl.LazyFrame) -> None:
    _df, result_dict = temporal_coverage(
        temporal_coverage_df,
        expected_start="1970-01-01T00:00:00",
        expected_end="1970-01-01T00:00:10",
        threshold=80.0,
    )
    qualification_assert(result_dict, TEMPORAL_COVERAGE, 100.0, True)


def test_fail(temporal_coverage_df: pl.LazyFrame) -> None:
    _df, result_dict = temporal_coverage(
        temporal_coverage_df,
        expected_start=datetime(1970, 1, 1, 0, 0, 0),
        expected_end=datetime(1970, 1, 1, 0, 0, 20),
        threshold=80.0,
    )
    qualification_assert(result_dict, TEMPORAL_COVERAGE, 50.0, False)


def test_invalid_expected_range(temporal_coverage_df: pl.LazyFrame) -> None:
    with pytest.raises(ValueError, match="expected_end must be after expected_start"):
        temporal_coverage(
            temporal_coverage_df,
            expected_start="1970-01-01T00:00:10",
            expected_end="1970-01-01T00:00:10",
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            datetime(1970, 1, 1, 0, 0, 10),
            datetime(1970, 1, 1, 0, 0, 10, tzinfo=UTC),
        ),
        (
            datetime(1970, 1, 1, 2, 0, 10, tzinfo=timezone(timedelta(hours=2))),
            datetime(1970, 1, 1, 0, 0, 10, tzinfo=UTC),
        ),
    ],
)
def test_ensure_utc(value: datetime, expected: datetime) -> None:
    assert _ensure_utc(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            datetime(1970, 1, 1, 0, 0, 10),
            datetime(1970, 1, 1, 0, 0, 10, tzinfo=UTC),
        ),
        (
            "1970-01-01T00:00:10",
            datetime(1970, 1, 1, 0, 0, 10, tzinfo=UTC),
        ),
        (
            "1970-01-01T02:00:10+02:00",
            datetime(1970, 1, 1, 0, 0, 10, tzinfo=UTC),
        ),
    ],
)
def test_parse_datetime(value: datetime | str, expected: datetime) -> None:
    assert _parse_datetime(value, "expected_start") == expected


def test_parse_datetime_invalid_string() -> None:
    with pytest.raises(ValueError, match="expected_start must be a datetime or ISO 8601 string"):
        _parse_datetime("not-a-datetime", "expected_start")


def test_parse_datetime_invalid_type() -> None:
    with pytest.raises(TypeError, match="expected_start must be a datetime or ISO 8601 string"):
        _parse_datetime(1.5, "expected_start")
