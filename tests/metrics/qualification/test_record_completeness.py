"""."""

import polars as pl

from omega_prime import Recording
from omega_prime.metrics.qualification.record_completeness import RECORD_COMPLETENESS, record_completeness

from .conftest import qualification_assert


def test_pass(rec: Recording) -> None:
    _df, result = record_completeness(rec.df)
    qualification_assert(result, RECORD_COMPLETENESS, 100.0, True)


def test_fail(rec: Recording) -> None:
    df = rec.df.with_columns(pl.lit(None).alias("x"))
    _df, result = record_completeness(df)
    qualification_assert(result, RECORD_COMPLETENESS, 95.0, False)


def test_pass_lt_100(rec: Recording) -> None:
    df = rec.df.with_columns(
        [
            pl.when(pl.arange(0, rec.df.height).is_in([0, 1, 2, 4])).then(None).otherwise(pl.col(col)).alias(col)
            for col in ["x", "y"]
        ]
    )

    _df, result = record_completeness(df)
    qualification_assert(result, RECORD_COMPLETENESS, 99.95391705069125, True)
