"""."""

from collections.abc import Sequence

import polars as pl
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT


TARGET_AREA_COVERAGE = "target_area_coverage"


@metric(computes_properties=[TARGET_AREA_COVERAGE])
def target_area_coverage(
    df: pl.LazyFrame,
    /,
    expected_area_coords: Sequence[tuple[float, float]],
    threshold: float = 80.0,
    expected_area_crs: str | CRS | None = None,
    proj_string: str | None = None,
) -> QRT:
    if len(expected_area_coords) < 3:
        raise ValueError("expected_area_coords must contain at least three coordinate pairs")

    transformed_coords = _transform_expected_coords(expected_area_coords, expected_area_crs, proj_string)
    expected_polygon = Polygon(transformed_coords)
    if expected_polygon.is_empty or not expected_polygon.is_valid:
        raise ValueError("expected_area_coords does not form a valid polygon")

    bounds = df.select(
        pl.min("x").alias("min_x"),
        pl.max("x").alias("max_x"),
        pl.min("y").alias("min_y"),
        pl.max("y").alias("max_y"),
    ).collect()

    if bounds.height == 0:
        dataset_polygon = Polygon()
        dataset_area = 0.0
        intersection_area = 0.0
    else:
        min_x, max_x, min_y, max_y = bounds.row(0)
        dataset_polygon = box(min_x, min_y, max_x, max_y)
        dataset_area = dataset_polygon.area
        intersection_area = dataset_polygon.intersection(expected_polygon).area

    expected_area = expected_polygon.area
    union_area = dataset_area + expected_area - intersection_area
    if union_area > 0:
        coverage = (intersection_area / union_area) * 100.0
    else:
        coverage = 100.0 if expected_area == 0 else 0.0

    status = PASS if coverage >= threshold else FAIL

    summary = pl.DataFrame(
        {
            "dataset_area": [dataset_area],
            "expected_area": [expected_area],
            "intersection_area": [intersection_area],
            TARGET_AREA_COVERAGE: [coverage],
            "threshold": [threshold],
            STATUS: [status],
        }
    ).lazy()

    return df, {TARGET_AREA_COVERAGE: summary}


def _transform_expected_coords(
    expected_coords: Sequence[tuple[float, float]],
    expected_area_crs: str | CRS | None,
    proj_string: str | None,
) -> list[tuple[float, float]]:
    if expected_area_crs is None:
        return list(expected_coords)
    if proj_string is None:
        raise ValueError("proj_string is required to transform expected_area_coords")

    dataset_crs = CRS.from_proj4(proj_string)
    source_crs = CRS.from_user_input(expected_area_crs)
    if source_crs == dataset_crs:
        return list(expected_coords)

    transformer = Transformer.from_crs(source_crs, dataset_crs, always_xy=True)
    xs, ys = zip(*expected_coords)
    return list(zip(*transformer.transform(xs, ys)))
