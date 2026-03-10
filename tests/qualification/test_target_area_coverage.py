"""."""

import math

import polars as pl
import pytest

from omega_prime.qualification.target_area_coverage import (
    _apply_proj_offset,
    _transform_expected_coords,
    target_area_coverage,
    TARGET_AREA_COVERAGE,
)

from .conftest import qualification_assert


EXPECTED_COORDS = [
    (0.0, 0.0),
    (10.0, 0.0),
    (10.0, 10.0),
    (0.0, 10.0),
]

PROJ_UTM32N = "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs"
EXPECTED_COORDS_WGS84 = [
    (6.050307, 50.779819),
    (6.050280, 50.779298),
    (6.050986, 50.779307),
    (6.050927, 50.779789),
]


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


def test_too_few_expected_area_coords() -> None:
    df = pl.DataFrame({"x": [0.0], "y": [0.0]}).lazy()

    with pytest.raises(
        ValueError,
        match="expected_area_coords must contain at least three coordinate pairs",
    ):
        target_area_coverage(
            df,
            expected_area_coords=[(0.0, 0.0), (1.0, 1.0)],
        )


def test_invalid_expected_area_polygon() -> None:
    df = pl.DataFrame({"x": [0.0], "y": [0.0]}).lazy()

    with pytest.raises(
        ValueError,
        match="expected_area_coords does not form a valid polygon",
    ):
        target_area_coverage(
            df,
            expected_area_coords=[
                (0.0, 0.0),
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ],
        )


def test_transform_expected_coords_pass() -> None:
    expected_coords_utm32n = [
        (292061.502, 5629488.771),
        (292057.287, 5629430.927),
        (292107.088, 5629429.941),
        (292105.068, 5629483.691),
    ]

    transformed_coords = _transform_expected_coords(
        EXPECTED_COORDS_WGS84,
        expected_area_crs="EPSG:4326",
        proj_string=PROJ_UTM32N,
        proj_offset=None,
    )

    for transformed, expected in zip(transformed_coords, expected_coords_utm32n, strict=True):
        assert transformed[0] == pytest.approx(expected[0])
        assert transformed[1] == pytest.approx(expected[1])


def test_apply_proj_offset_pass() -> None:
    adjusted_coords = _apply_proj_offset(
        [(1.0, 0.0), (2.0, 0.0), (1.0, 1.0)],
        (1.0, 0.0, math.pi / 2),
    )

    assert adjusted_coords[0] == pytest.approx((0.0, 0.0))
    assert adjusted_coords[1] == pytest.approx((0.0, -1.0))
    assert adjusted_coords[2] == pytest.approx((1.0, 0.0))


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
        expected_area_crs="EPSG:4326",
        proj_string=PROJ_UTM32N,
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
        expected_area_crs="EPSG:4326",
        proj_string=PROJ_UTM32N,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 50.0, False)
