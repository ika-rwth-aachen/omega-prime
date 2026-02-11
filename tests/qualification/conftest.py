"""."""

from pathlib import Path

import pytest

import omega_prime
from omega_prime.qualification.common import STATUS, PASS, FAIL
import polars as pl


@pytest.fixture()
def rec(files_dir: Path):
    return omega_prime.Recording.from_file(str(files_dir / "pedestrian.osi"))


def qualification_assert(qd: dict[str, pl.LazyFrame], metric_name: str, expected_value: float, is_pass: bool) -> None:
    result_df = qd[metric_name].collect()
    assert result_df[metric_name][0] == pytest.approx(expected_value)
    assert result_df[STATUS][0] == PASS if is_pass else FAIL
