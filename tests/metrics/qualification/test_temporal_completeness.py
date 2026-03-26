"""."""

import pytest
import polars as pl

from omega_prime import Recording

from omega_prime.metrics.qualification.temporal_completeness import TEMPORAL_COMPLETENESS, temporal_completeness

from .conftest import qualification_assert


@pytest.fixture()
def temporal_df_pass() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "idx": [1, 1, 1, 2, 2, 2],
            "total_nanos": [
                0,
                100_000_000,
                200_000_000,
                0,
                100_000_000,
                200_000_000,
            ],
        }
    ).lazy()


@pytest.fixture()
def temporal_df_mixed() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "idx": [1, 1, 1, 2, 2, 2],
            "total_nanos": [
                0,
                100_000_000,
                200_000_000,
                0,
                200_000_000,
                400_000_000,
            ],
        }
    ).lazy()


def test_pass(temporal_df_pass: pl.LazyFrame) -> None:
    _df, result = temporal_completeness(temporal_df_pass, expected_frequency=10.0)
    qualification_assert(result, TEMPORAL_COMPLETENESS, 100.0, True)


def test_fail(temporal_df_mixed: pl.LazyFrame) -> None:
    _df, result = temporal_completeness(temporal_df_mixed, expected_frequency=10.0)
    qualification_assert(result, TEMPORAL_COMPLETENESS, 50.0, False)


def test_record(rec: Recording) -> None:
    df, result = temporal_completeness(rec.df.lazy(), expected_frequency=30.0)
    qualification_assert(result, TEMPORAL_COMPLETENESS, 100.0, True)


def test_empty_df() -> None:
    empty_df = pl.DataFrame(
        {"idx": [], "total_nanos": []},
        schema={"idx": pl.Int64, "total_nanos": pl.Int64},
    ).lazy()

    _df, result = temporal_completeness(empty_df, expected_frequency=10.0)

    qualification_assert(result, TEMPORAL_COMPLETENESS, 0.0, False)


@pytest.mark.parametrize("expected_frequency", [0.0, -1.0])
def test_invalid_expected_frequency(temporal_df_pass: pl.LazyFrame, expected_frequency: float) -> None:
    with pytest.raises(ValueError, match="expected_frequency must be > 0"):
        temporal_completeness(temporal_df_pass, expected_frequency=expected_frequency)
