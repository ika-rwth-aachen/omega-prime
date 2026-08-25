"""."""

import pytest


from ..conftest import qualification_assert
from omega_prime.metrics.qualification.class_completeness import (
    CLASS_COMPLETENESS,
    ROLE_COMPLETENESS,
    SUBTYPE_COMPLETENESS,
    TYPE_COMPLETENESS,
)
import polars as pl


def assert_class_completeness(qd: dict[str, pl.LazyFrame], expected_values: dict[str, float], is_pass: bool) -> None:
    assert_subfield_completeness(qd, TYPE_COMPLETENESS, expected_values[TYPE_COMPLETENESS])
    assert_subfield_completeness(qd, SUBTYPE_COMPLETENESS, expected_values[SUBTYPE_COMPLETENESS])
    assert_subfield_completeness(qd, ROLE_COMPLETENESS, expected_values[ROLE_COMPLETENESS])
    class_completeness_expected_value = min(
        expected_values[TYPE_COMPLETENESS],
        expected_values[SUBTYPE_COMPLETENESS],
        expected_values[ROLE_COMPLETENESS],
    )
    qualification_assert(qd, CLASS_COMPLETENESS, class_completeness_expected_value, is_pass)


def assert_subfield_completeness(qd: dict[str, pl.LazyFrame], field_name: str, expected_value: float) -> None:
    result_df = qd[CLASS_COMPLETENESS].collect()
    assert result_df[field_name][0] == pytest.approx(expected_value)
