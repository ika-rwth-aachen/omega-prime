"""."""

import math

import pytest

from omega_prime.qualification.target_area_coverage import _transform_expected_coords

from .conftest import EXPECTED_COORDS, EXPECTED_COORDS_WGS84, EXPECTED_COORDS_UTM32N, PROJ_UTM32N


def test_transform_expected_coords_pass() -> None:
    transformed_coords = _transform_expected_coords(
        EXPECTED_COORDS_WGS84,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4=PROJ_UTM32N,
        dataset_frame_offset=None,
    )

    for transformed, expected in zip(transformed_coords, EXPECTED_COORDS_UTM32N, strict=True):
        assert transformed[0] == pytest.approx(expected[0])
        assert transformed[1] == pytest.approx(expected[1])


def test_transform_expected_coords_with_matching_crs() -> None:
    transformed_coords = _transform_expected_coords(
        EXPECTED_COORDS_UTM32N,
        expected_area_source_crs="EPSG:32632",
        dataset_proj4=PROJ_UTM32N,
        dataset_frame_offset=None,
    )

    assert transformed_coords == EXPECTED_COORDS_UTM32N


def test_transform_expected_coords_with_matching_crs_and_offset() -> None:
    transformed_coords = _transform_expected_coords(
        [(1.0, 0.0), (2.0, 0.0), (1.0, 1.0)],
        expected_area_source_crs="EPSG:32632",
        dataset_proj4=PROJ_UTM32N,
        dataset_frame_offset=(1.0, 0.0, math.pi / 2),
    )

    assert transformed_coords[0] == pytest.approx((0.0, 0.0))
    assert transformed_coords[1] == pytest.approx((0.0, -1.0))
    assert transformed_coords[2] == pytest.approx((1.0, 0.0))


def test_transform_expected_coords_without_expected_area_source_crs_fails() -> None:
    with pytest.raises(
        ValueError,
        match="expected_area_source_crs is required when dataset_proj4 or dataset_frame_offset is provided",
    ):
        _transform_expected_coords(
            EXPECTED_COORDS,
            expected_area_source_crs=None,
            dataset_proj4=PROJ_UTM32N,
            dataset_frame_offset=(1.0, 2.0, math.pi / 4),
        )


def test_transform_expected_coords_without_dataset_proj4() -> None:
    with pytest.raises(
        ValueError,
        match="dataset_proj4 is required to transform expected_area_coords",
    ):
        _transform_expected_coords(
            EXPECTED_COORDS_WGS84,
            expected_area_source_crs="EPSG:4326",
            dataset_proj4=None,
            dataset_frame_offset=None,
        )
