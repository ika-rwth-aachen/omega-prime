"""."""

from datetime import UTC, datetime, timedelta, timezone

import polars as pl
import pytest

from omega_prime.qualification.temporal_coverage import _ensure_utc, temporal_coverage, TEMPORAL_COVERAGE

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
