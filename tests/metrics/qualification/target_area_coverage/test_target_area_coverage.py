"""."""

import polars as pl
import pytest
from shapely.geometry import MultiPoint, Polygon

from omega_prime.metrics.qualification.target_area_coverage import (
    TARGET_AREA_COVERAGE,
    transform_expected_coords,
    target_area_coverage,
)

from ..conftest import qualification_assert


@pytest.fixture
def transformed_expected_coords(
    expected_coords_wgs84: list[tuple[float, float]],
    proj_utm32n: str,
) -> list[tuple[float, float]]:
    return transform_expected_coords(
        expected_coords_wgs84,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4=proj_utm32n,
    )


def test_pass(expected_coords_local: list[tuple[float, float]]) -> None:
    df = pl.DataFrame(
        {
            "x": [0.0, 10.0, 0.0, 10.0],
            "y": [0.0, 0.0, 10.0, 10.0],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=expected_coords_local,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 100.0, True)


def test_fail(expected_coords_local: list[tuple[float, float]]) -> None:
    df = pl.DataFrame(
        {
            "x": [20.0, 30.0],
            "y": [20.0, 30.0],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=expected_coords_local,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 0.0, False)


def test_with_projection_pass(
    expected_coords_wgs84: list[tuple[float, float]],
    proj_utm32n: str,
    transformed_expected_coords: list[tuple[float, float]],
) -> None:
    df = pl.DataFrame(
        {
            "x": [coord[0] for coord in transformed_expected_coords],
            "y": [coord[1] for coord in transformed_expected_coords],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=expected_coords_wgs84,
        threshold=100.0,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4=proj_utm32n,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 100.0, True)


def test_with_projection_partial_overlap(
    expected_coords_wgs84: list[tuple[float, float]],
    proj_utm32n: str,
    transformed_expected_coords: list[tuple[float, float]],
) -> None:
    covered_coords = transformed_expected_coords[:3]
    df = pl.DataFrame(
        {
            "x": [coord[0] for coord in covered_coords],
            "y": [coord[1] for coord in covered_coords],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=expected_coords_wgs84,
        threshold=100.0,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4=proj_utm32n,
    )
    expected_polygon = Polygon(transformed_expected_coords)
    expected_coverage = MultiPoint(covered_coords).convex_hull.intersection(expected_polygon).area * 100.0
    expected_coverage /= expected_polygon.area

    result_df = result_dict[TARGET_AREA_COVERAGE].collect()
    assert result_df[TARGET_AREA_COVERAGE][0] == pytest.approx(expected_coverage)
    assert result_df["status"][0] == "fail"


def test_ignores_coverage_outside_expected_area(expected_coords_local: list[tuple[float, float]]) -> None:
    df = pl.DataFrame(
        {
            "x": [-5.0, 15.0, 15.0, -5.0],
            "y": [-5.0, -5.0, 15.0, 15.0],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=expected_coords_local,
        threshold=100.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 100.0, True)


def test_empty_df(expected_coords_local: list[tuple[float, float]]) -> None:
    df = pl.DataFrame(
        {
            "x": [],
            "y": [],
        },
        schema={"x": pl.Float64, "y": pl.Float64},
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=expected_coords_local,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 0.0, False)


def test_degenerate_covered_area(expected_coords_local: list[tuple[float, float]]) -> None:
    df = pl.DataFrame(
        {
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=expected_coords_local,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 0.0, False)
