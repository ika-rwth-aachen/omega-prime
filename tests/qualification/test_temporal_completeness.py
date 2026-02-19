"""."""

import pytest
import polars as pl

from omega_prime.qualification.temporal_completeness import temporal_completeness, TEMPORAL_COMPLETENESS

from .conftest import qualification_assert

@pytest.fixture()
def temporal_df_pass() -> pl.DataFrame:
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
    )

@pytest.fixture()
def temporal_df_mixed() -> pl.DataFrame:
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
    )

def test_pass(temporal_df_pass: pl.DataFrame) -> None:
    _df, result = temporal_completeness(
        temporal_df_pass,
        expected_frequency=10.0
    )
    qualification_assert(result, TEMPORAL_COMPLETENESS, 100.0, True)


def test_fail(temporal_df_mixed: pl.DataFrame) -> None:
    _df, result = temporal_completeness(
        temporal_df_mixed,
        expected_frequency=10.0
    )
    qualification_assert(result, TEMPORAL_COMPLETENESS, 50.0, False)
