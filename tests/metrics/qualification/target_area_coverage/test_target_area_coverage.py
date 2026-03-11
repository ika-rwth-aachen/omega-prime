"""."""

import polars as pl

from omega_prime.qualification.target_area_coverage import target_area_coverage, TARGET_AREA_COVERAGE

from .conftest import EXPECTED_COORDS, EXPECTED_COORDS_WGS84, PROJ_UTM32N
from ..conftest import qualification_assert


def test_pass() -> None:
    df = pl.DataFrame(
        {
            "x": [0.0, 10.0, 0.0, 10.0],
            "y": [0.0, 0.0, 10.0, 10.0],
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
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 0.0, False)


def test_target_area_coverage_with_projection_pass() -> None:
    df = pl.DataFrame(
        {
            "x": [
                292080.684,
                292089.717,
                292084.573,
            ],
            "y": [
                5629461.727,
                5629454.574,
                5629460.124,
            ],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS_WGS84,
        threshold=100.0,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4=PROJ_UTM32N,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 100.0, True)


def test_target_area_coverage_with_projection_fail() -> None:
    df = pl.DataFrame(
        {
            "x": [
                292019.412,
                292183.911,
                292087.379,
                292081.964,
            ],
            "y": [
                5629435.334,
                5629509.941,
                5629457.897,
                5629454.884,
            ],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS_WGS84,
        threshold=100.0,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4=PROJ_UTM32N,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 50.0, False)
