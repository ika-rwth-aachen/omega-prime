"""."""

from collections.abc import Sequence
import math

import polars as pl
from pyproj import CRS, Transformer
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT


TARGET_AREA_COVERAGE = "target_area_coverage"


@metric(computes_properties=[TARGET_AREA_COVERAGE])
def target_area_coverage(
    df: pl.LazyFrame,
    /,
    expected_area_coords: Sequence[tuple[float, float]],
    threshold: float = 80.0,
    expected_area_source_crs: str | CRS | None = None,
    dataset_proj4: str | None = None,
    dataset_frame_offset: tuple[float, float, float] | None = None,
) -> QRT:
    if len(expected_area_coords) < 3:
        raise ValueError("expected_area_coords must contain at least three coordinate pairs")

    transformed_coords = _transform_expected_coords(
        expected_area_coords,
        expected_area_source_crs,
        dataset_proj4,
        dataset_frame_offset,
    )
    expected_polygon = Polygon(transformed_coords)
    if expected_polygon.is_empty or not expected_polygon.is_valid:
        raise ValueError("expected_area_coords does not form a valid polygon")

    point_df = df.select("x", "y").collect()
    total_points = point_df.height
    points_inside = 0
    if total_points > 0:
        prepared_polygon = prep(expected_polygon)
        points_inside = sum(prepared_polygon.covers(Point(x, y)) for x, y in point_df.iter_rows())

    coverage = points_inside * 100.0 / total_points if total_points > 0 else 100.0

    status = PASS if coverage >= threshold else FAIL

    summary = pl.DataFrame(
        {
            TARGET_AREA_COVERAGE: [coverage],
            STATUS: [status],
        }
    ).lazy()

    return df, {TARGET_AREA_COVERAGE: summary}


def _transform_expected_coords(
    expected_coords: Sequence[tuple[float, float]],
    expected_area_source_crs: str | CRS | None,
    dataset_proj4: str | None,
    dataset_frame_offset: tuple[float, float, float] | None,
) -> list[tuple[float, float]]:
    if expected_area_source_crs is None:
        if dataset_proj4 is not None or dataset_frame_offset is not None:
            raise ValueError(
                "expected_area_source_crs is required when dataset_proj4 or dataset_frame_offset is provided"
            )
        return list(expected_coords)
    if dataset_proj4 is None:
        raise ValueError("dataset_proj4 is required to transform expected_area_coords")

    dataset_crs = CRS.from_proj4(dataset_proj4)
    source_crs = CRS.from_user_input(expected_area_source_crs)
    if source_crs == dataset_crs:
        transformed = list(expected_coords)
    else:
        transformer = Transformer.from_crs(source_crs, dataset_crs, always_xy=True)
        xs, ys = zip(*expected_coords)
        transformed = list(zip(*transformer.transform(xs, ys)))
    if dataset_frame_offset is None:
        return transformed

    return _apply_proj_offset(transformed, dataset_frame_offset)


def _apply_proj_offset(
    coords: Sequence[tuple[float, float]],
    dataset_frame_offset: tuple[float, float, float],
) -> list[tuple[float, float]]:
    offset_x, offset_y, offset_yaw = dataset_frame_offset
    cos_yaw = math.cos(-offset_yaw)
    sin_yaw = math.sin(-offset_yaw)
    adjusted: list[tuple[float, float]] = []
    for x, y in coords:
        x_shift = x - offset_x
        y_shift = y - offset_y
        x_rot = cos_yaw * x_shift - sin_yaw * y_shift
        y_rot = sin_yaw * x_shift + cos_yaw * y_shift
        adjusted.append((x_rot, y_rot))
    return adjusted
