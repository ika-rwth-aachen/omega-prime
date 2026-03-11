"""."""

import pytest
import polars as pl

from omega_prime import Recording
from omega_prime.metrics.qualification.duplicate_record_rate import DUPLICATE_RECORD_RATE, duplicate_record_rate

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
    _df, result_dict = duplicate_record_rate(duplicate_df_pass)
    qualification_assert(result_dict, DUPLICATE_RECORD_RATE, 0.0, True)


def test_fail(duplicate_df_fail) -> None:
    _df, result_dict = duplicate_record_rate(duplicate_df_fail)
    qualification_assert(result_dict, DUPLICATE_RECORD_RATE, 66.66666666666667, False)


def test_record_pass(rec: Recording) -> None:
    _df, result_dict = duplicate_record_rate(rec.df.lazy())
    qualification_assert(result_dict, DUPLICATE_RECORD_RATE, 0.0, True)
