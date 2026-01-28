"""."""

from datetime import datetime

import pytest

from omega_prime.qualification.temporal_coverage import temporal_coverage


@pytest.fixture()
def temporal_coverage_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "total_nanos": [0, 10_000_000_000],
        }
    )

def test_pass(temporal_coverage_df) -> None:
    _df, result_dict = temporal_coverage(
        temporal_coverage_df,
        expected_start="1970-01-01T00:00:00",
        expected_end="1970-01-01T00:00:10",
        threshold=80.0,
    )
    summary = result_dict["temporal_coverage"].collect()
    assert summary["temporal_coverage"][0] == pytest.approx(100.0)
    assert summary["status"][0] == "pass"


def test_fail(temporal_coverage_df) -> None:
    _df, result_dict = temporal_coverage(
        temporal_coverage_df,
        expected_start=datetime(1970, 1, 1, 0, 0, 0),
        expected_end=datetime(1970, 1, 1, 0, 0, 20),
        threshold=80.0,
    )
    summary = result_dict["temporal_coverage"].collect()
    assert summary["temporal_coverage"][0] == pytest.approx(50.0)
    assert summary["status"][0] == "fail"
