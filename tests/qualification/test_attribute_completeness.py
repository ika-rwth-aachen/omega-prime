"""."""

import pytest

from omega_prime import Recording
from omega_prime.qualification.common import STATUS, PASS, FAIL
from omega_prime.qualification.attribute_completeness import attribute_completeness, ATTRIBUTE_COMPLETENESS
from omega_prime.schemas import polars_schema


def test_pass(rec: Recording) -> None:
    _df, result = attribute_completeness(rec.df)
    result_df = result[ATTRIBUTE_COMPLETENESS].collect()
    assert result_df[ATTRIBUTE_COMPLETENESS][0] == pytest.approx(100.0)
    assert result_df[STATUS][0] == PASS


def test_fail(rec: Recording) -> None:
    for column_to_drop in polars_schema:
        incomplete_df = rec.df.drop(column_to_drop)
        _df, result = attribute_completeness(incomplete_df)
        result_df = result[ATTRIBUTE_COMPLETENESS].collect()
        assert result_df[ATTRIBUTE_COMPLETENESS][0] == pytest.approx(95.0)
        assert result_df[STATUS][0] == FAIL
