"""."""

import polars as pl

from omega_prime.qualification.target_area_coverage import target_area_coverage, TARGET_AREA_COVERAGE

from .conftest import qualification_assert


EXPECTED_COORDS = [
    (0.0, 0.0),
    (10.0, 0.0),
    (10.0, 10.0),
    (0.0, 10.0),
]


def test_pass() -> None:
    df = pl.DataFrame(
        {
            "x": [0.0, 10.0, 0.0, 10.0],
            "y": [0.0, 0.0, 10.0, 10.0],
            "total_nanos": [0, 0, 1, 1],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 100.0, True)


def test_fail() -> None:
    df = pl.DataFrame(
        {
            "x": [20.0, 30.0],
            "y": [20.0, 30.0],
            "total_nanos": [0, 1],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 0.0, False)
