"""."""

import polars as pl
import pytest

from omega_prime.metrics.qualification.common import FAIL, PASS, STATUS


def qualification_assert(qd: dict[str, pl.LazyFrame], metric_name: str, expected_value: float, is_pass: bool) -> None:
    result_df = qd[metric_name].collect()
    assert result_df[metric_name][0] == pytest.approx(expected_value)
    expected_status = PASS if is_pass else FAIL
    assert result_df[STATUS][0] == expected_status
