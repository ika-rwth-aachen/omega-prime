"""."""

import math

import pytest

from omega_prime.metrics.qualification.target_area_coverage import transform_expected_coords


def test_pass(
    expected_coords_wgs84: list[tuple[float, float]],
    expected_coords_utm32n: list[tuple[float, float]],
    proj_utm32n: str,
) -> None:
    transformed_coords = transform_expected_coords(
        expected_coords_wgs84,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4=proj_utm32n,
    )

    for transformed, expected in zip(transformed_coords, expected_coords_utm32n, strict=True):
        assert transformed[0] == pytest.approx(expected[0])
        assert transformed[1] == pytest.approx(expected[1])


def test_with_matching_crs(
    expected_coords_utm32n: list[tuple[float, float]],
    proj_utm32n: str,
) -> None:
    transformed_coords = transform_expected_coords(
        expected_coords_utm32n,
        expected_area_source_crs="EPSG:32632",
        dataset_proj4=proj_utm32n,
    )

    assert transformed_coords == expected_coords_utm32n


def test_accepts_dataset_crs_user_input(
    expected_coords_wgs84: list[tuple[float, float]],
    expected_coords_utm32n: list[tuple[float, float]],
) -> None:
    transformed_coords = transform_expected_coords(
        expected_coords_wgs84,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4="EPSG:32632",
    )

    for transformed, expected in zip(transformed_coords, expected_coords_utm32n, strict=True):
        assert transformed[0] == pytest.approx(expected[0])
        assert transformed[1] == pytest.approx(expected[1])


def test_with_matching_crs_and_offset(proj_utm32n: str) -> None:
    transformed_coords = transform_expected_coords(
        [(1.0, 0.0), (2.0, 0.0), (1.0, 1.0)],
        expected_area_source_crs="EPSG:32632",
        dataset_proj4=proj_utm32n,
        dataset_frame_offset=(1.0, 0.0, math.pi / 2),
    )

    assert transformed_coords[0] == pytest.approx((0.0, 0.0))
    assert transformed_coords[1] == pytest.approx((0.0, -1.0))
    assert transformed_coords[2] == pytest.approx((1.0, 0.0))


def test_without_expected_area_source_crs_fails(
    expected_coords_local: list[tuple[float, float]],
    proj_utm32n: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="expected_area_source_crs is required when dataset_proj4 or dataset_frame_offset is provided",
    ):
        transform_expected_coords(
            expected_coords_local,
            expected_area_source_crs="",
            dataset_proj4=proj_utm32n,
            dataset_frame_offset=(1.0, 2.0, math.pi / 4),
        )


def test_without_dataset_proj4(
    expected_coords_wgs84: list[tuple[float, float]],
) -> None:
    with pytest.raises(
        ValueError,
        match="dataset_proj4 is required to transform expected_area_coords",
    ):
        transform_expected_coords(
            expected_coords_wgs84,
            expected_area_source_crs="EPSG:4326",
            dataset_proj4="",
        )
