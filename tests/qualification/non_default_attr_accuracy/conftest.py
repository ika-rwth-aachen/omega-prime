"""."""

import pytest

from omega_prime.qualification.common import STATUS, PASS, FAIL
from omega_prime.qualification.non_default_attr_accuracy import NON_DEFAULT_ATTRIBUTES_ACCURACY
import polars as pl


def this_assert(qd: dict[str, pl.LazyFrame], expected_value: float, is_pass: bool) -> None:
    result_df = qd[NON_DEFAULT_ATTRIBUTES_ACCURACY].collect()
    assert result_df[NON_DEFAULT_ATTRIBUTES_ACCURACY][0] == pytest.approx(expected_value)
    assert result_df[STATUS][0] == PASS if is_pass else FAIL
