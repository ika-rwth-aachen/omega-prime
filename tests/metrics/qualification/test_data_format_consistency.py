"""."""

import polars as pl

from omega_prime import Recording
from omega_prime.metrics.qualification.data_format_consistency import (
    DATA_FORMAT_CONSISTENCY,
    data_format_consistency,
)

from .conftest import qualification_assert


def test_pass(rec: Recording) -> None:
    _df, result = data_format_consistency(rec.df)
    qualification_assert(result, DATA_FORMAT_CONSISTENCY, 100.0, True)


def test_fail_for_out_of_range_type() -> None:
    df = pl.DataFrame({"type": [0, 5, 4]}).lazy()
    _df, result = data_format_consistency(df)
    qualification_assert(result, DATA_FORMAT_CONSISTENCY, 66.66666666666667, False)


def test_fail_for_wrong_dtype() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]}).lazy()
    _df, result = data_format_consistency(df)
    qualification_assert(result, DATA_FORMAT_CONSISTENCY, 0.0, False)


def test_pass_for_irrelevant_columns_only() -> None:
    df = pl.DataFrame({"bogus": [1, 2, 3]}).lazy()
    _df, result = data_format_consistency(df)
    qualification_assert(result, DATA_FORMAT_CONSISTENCY, 100.0, True)
