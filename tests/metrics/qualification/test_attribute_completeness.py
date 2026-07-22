"""."""

from omega_prime import Recording
from omega_prime.metrics.qualification.attribute_completeness import ATTRIBUTE_COMPLETENESS, attribute_completeness
from omega_prime.schemas import polars_schema

from .conftest import qualification_assert


def test_pass(rec: Recording) -> None:
    _df, result = attribute_completeness(rec.df)
    qualification_assert(result, ATTRIBUTE_COMPLETENESS, 100.0, True)


def test_fail(rec: Recording) -> None:
    for column_to_drop in polars_schema:
        incomplete_df = rec.df.drop(column_to_drop)
        _df, result = attribute_completeness(incomplete_df)
        qualification_assert(result, ATTRIBUTE_COMPLETENESS, 95.0, False)
