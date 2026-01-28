"""."""

from collections.abc import Sequence
import math
from typing import Any

import polars as pl
from shapely.geometry import Polygon, Point, box
from shapely.prepared import prep
from pyproj import CRS, Transformer

from omega_prime import Recording

from ..metrics import metric


@metric(computes_properties=["target_area_coverage", "target_area_frame_coverage"])
def target_area_coverage(
    df: pl.LazyFrame | pl.DataFrame,
    /,
    expected_area_coords: Sequence[tuple[float, float]],
    threshold: float = 80.0,
    frame_threshold: float | None = None,
    frame_column: str = "total_nanos",
    expected_area_crs: str | CRS | None = "EPSG:4326",
    recording: Recording | None = None,
):
    if len(expected_area_coords) < 3:
        raise ValueError("expected_area_coords must contain at least three coordinate pairs")

    transformed_coords = _transform_expected_coords(expected_area_coords, recording, expected_area_crs)
    expected_polygon = Polygon(transformed_coords)
    if expected_polygon.is_empty or not expected_polygon.is_valid:
        raise ValueError("expected_area_coords does not form a valid polygon")

    schema = df.collect_schema()
    df_columns = schema.names()

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

    status = "pass" if coverage >= threshold else "fail"

    frame_records: list[dict[str, Any]] = []
    total_frames = 0
    frames_inside = 0
    frame_coverage_percent = 0.0
    frame_status: str | None = None

    if frame_column in df_columns:
        frame_positions = df.select(frame_column, "x", "y").collect()
        if frame_positions.height > 0:
            prepared_expected = prep(expected_polygon)
            frame_stats: dict[Any, dict[str, int]] = {}
            for row in frame_positions.iter_rows(named=True):
                frame_val = row[frame_column]
                inside = prepared_expected.covers(Point(row["x"], row["y"]))
                stats = frame_stats.setdefault(frame_val, {"total": 0, "inside": 0})
                stats["total"] += 1
                if inside:
                    stats["inside"] += 1

            total_frames = len(frame_stats)
            for frame_val, stats in frame_stats.items():
                inside_flag = stats["inside"] > 0
                fraction_inside_percent = (
                    stats["inside"] / stats["total"] * 100.0 if stats["total"] else 0.0
                )
                if inside_flag:
                    frames_inside += 1
                frame_records.append(
                    {
                        frame_column: frame_val,
                        "total_objects": stats["total"],
                        "objects_inside": stats["inside"],
                        "fraction_inside_percent": fraction_inside_percent,
                        "inside_expected_area": inside_flag,
                    }
                )

            if total_frames > 0:
                frame_coverage_percent = frames_inside / total_frames * 100.0
                if frame_threshold is not None:
                    frame_status = "pass" if frame_coverage_percent >= frame_threshold else "fail"

    summary = pl.DataFrame(
        {
            "dataset_area": [dataset_area],
            "expected_area": [expected_area],
            "intersection_area": [intersection_area],
            "target_area_coverage": [coverage],
            "threshold": [threshold],
            "status": [status],
            "total_frames": [total_frames],
            "frames_inside": [frames_inside],
            "frame_coverage": [frame_coverage_percent],
            "frame_threshold": [frame_threshold],
            "frame_status": [frame_status],
        }
    ).lazy()

    if frame_records:
        frame_df = pl.DataFrame(frame_records).lazy()
    else:
        frame_dtype = schema.get(frame_column, pl.Float64)
        frame_df = pl.DataFrame(
            {
                frame_column: pl.Series(name=frame_column, values=[], dtype=frame_dtype),
                "total_objects": pl.Series(name="total_objects", values=[], dtype=pl.UInt32),
                "objects_inside": pl.Series(name="objects_inside", values=[], dtype=pl.UInt32),
                "fraction_inside_percent": pl.Series(name="fraction_inside_percent", values=[], dtype=pl.Float64),
                "inside_expected_area": pl.Series(name="inside_expected_area", values=[], dtype=pl.Boolean),
            }
        ).lazy()

    return df, {
        "target_area_coverage": summary,
        "target_area_frame_coverage": frame_df,
    }


def _transform_expected_coords(
    expected_coords: Sequence[tuple[float, float]],
    recording: Recording | None,
    expected_area_crs: str | CRS | None,
) -> list[tuple[float, float]]:
    if expected_area_crs is None:
        return list(expected_coords)
    if recording is None or len(recording.projections) == 0:
        raise ValueError("Recording with projection information is required to transform expected_area_coords")

    proj_info = recording.projections[0]
    proj_string = proj_info.get("proj_string")
    if not proj_string:
        raise ValueError("Recording projection information does not contain a valid proj_string")

    dataset_crs = CRS.from_proj4(proj_string)
    source_crs = CRS.from_user_input(expected_area_crs) if expected_area_crs is not None else dataset_crs
    if source_crs == dataset_crs:
        transformed = list(expected_coords)
    else:
        transformer = Transformer.from_crs(source_crs, dataset_crs, always_xy=True)
        xs, ys = zip(*expected_coords)
        transformed = list(zip(*transformer.transform(xs, ys)))

    offset = proj_info.get("offset")
    if offset is not None:
        cos_yaw = math.cos(-offset.yaw)
        sin_yaw = math.sin(-offset.yaw)
        adjusted = []
        for x, y in transformed:
            x_shift = x - offset.x
            y_shift = y - offset.y
            x_rot = cos_yaw * x_shift - sin_yaw * y_shift
            y_rot = sin_yaw * x_shift + cos_yaw * y_shift
            adjusted.append((x_rot, y_rot))
        return adjusted

    return transformed
