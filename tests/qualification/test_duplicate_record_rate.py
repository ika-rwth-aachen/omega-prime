"""."""

import pytest
import polars as pl

from omega_prime.qualification.duplicate_record_rate import (
    duplicate_record_rate,
    DUPLICATE_RECORD_RATE,
    DUPLICATE_RECORD_DUPLICATES,
)

from .conftest import qualification_assert


@pytest.fixture()
def duplicate_df_pass() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "idx": [1, 1, 2, 2],
            "total_nanos": [0, 1, 0, 1],
        }
    ).lazy()


@pytest.fixture()
def duplicate_df_fail() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "idx": [1, 1, 1, 2, 2, 2],
            "total_nanos": [0, 0, 0, 1, 1, 1],
        }
    ).lazy()


def test_pass(duplicate_df_pass) -> None:
    _df, result_dict = duplicate_record_rate(duplicate_df_pass, threshold=1.0)
    summary = result_dict[DUPLICATE_RECORD_RATE].collect()
    assert summary["total_records"][0] == 4
    assert summary["duplicate_records"][0] == 0
    qualification_assert(result_dict, DUPLICATE_RECORD_RATE, 0.0, True)


def test_fail(duplicate_df_fail) -> None:
    _df, result_dict = duplicate_record_rate(duplicate_df_fail, threshold=1.0)
    summary = result_dict[DUPLICATE_RECORD_RATE].collect()
    duplicates = result_dict[DUPLICATE_RECORD_DUPLICATES].collect()
    assert summary["total_records"][0] == 6
    assert summary["duplicate_records"][0] == 4
    qualification_assert(result_dict, DUPLICATE_RECORD_RATE, 66.66666666666667, False)
    assert duplicates["duplicate_records"].sum() == 4
