"""."""

import polars as pl
import pytest
from shapely.geometry import MultiPoint, Polygon

from omega_prime.metrics.qualification.target_area_coverage import (
    TARGET_AREA_COVERAGE,
    _transform_expected_coords,
    target_area_coverage,
)

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
    covered_coords = _transform_expected_coords(
        EXPECTED_COORDS_WGS84,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4=PROJ_UTM32N,
        dataset_frame_offset=None,
    )
    df = pl.DataFrame(
        {
            "x": [coord[0] for coord in covered_coords],
            "y": [coord[1] for coord in covered_coords],
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


def test_target_area_coverage_with_projection_partial_overlap() -> None:
    covered_coords = _transform_expected_coords(
        EXPECTED_COORDS_WGS84,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4=PROJ_UTM32N,
        dataset_frame_offset=None,
    )[:3]
    df = pl.DataFrame(
        {
            "x": [coord[0] for coord in covered_coords],
            "y": [coord[1] for coord in covered_coords],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS_WGS84,
        threshold=100.0,
        expected_area_source_crs="EPSG:4326",
        dataset_proj4=PROJ_UTM32N,
    )
    expected_polygon = Polygon(
        _transform_expected_coords(
            EXPECTED_COORDS_WGS84,
            expected_area_source_crs="EPSG:4326",
            dataset_proj4=PROJ_UTM32N,
            dataset_frame_offset=None,
        )
    )
    expected_coverage = MultiPoint(covered_coords).convex_hull.intersection(expected_polygon).area * 100.0
    expected_coverage /= expected_polygon.area

    result_df = result_dict[TARGET_AREA_COVERAGE].collect()
    assert result_df[TARGET_AREA_COVERAGE][0] == pytest.approx(expected_coverage)
    assert result_df["status"][0] == "fail"


def test_target_area_coverage_ignores_coverage_outside_expected_area() -> None:
    df = pl.DataFrame(
        {
            "x": [-5.0, 15.0, 15.0, -5.0],
            "y": [-5.0, -5.0, 15.0, 15.0],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS,
        threshold=100.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 100.0, True)


def test_target_area_coverage_empty_df() -> None:
    df = pl.DataFrame(
        {
            "x": [],
            "y": [],
        },
        schema={"x": pl.Float64, "y": pl.Float64},
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 0.0, False)


def test_target_area_coverage_degenerate_covered_area() -> None:
    df = pl.DataFrame(
        {
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 0.0, False)
