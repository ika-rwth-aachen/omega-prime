"""."""

from datetime import datetime

import polars as pl

from ..metrics import metric


@metric(computes_properties=["temporal_coverage"])
def temporal_coverage(
    df: pl.LazyFrame | pl.DataFrame,
    /,
    expected_start: datetime | str,
    expected_end: datetime | str,
    threshold: float = 80.0,
):
    start_dt = _parse_datetime(expected_start, "expected_start")
    end_dt = _parse_datetime(expected_end, "expected_end")
    if end_dt <= start_dt:
        raise ValueError("expected_end must be after expected_start")

    required_seconds = (end_dt - start_dt).total_seconds()
    row_count = df.select(pl.len().alias("row_count")).collect()[0, "row_count"]
    if row_count == 0:
        dataset_start = dataset_end = None
        coverage = 0.0
        overlap_seconds = 0.0
    else:
        nanos_bounds = df.select(
            pl.min("total_nanos").alias("min_nanos"),
            pl.max("total_nanos").alias("max_nanos"),
        ).collect()
        min_nanos = nanos_bounds[0, "min_nanos"]
        max_nanos = nanos_bounds[0, "max_nanos"]
        dataset_start = datetime.utcfromtimestamp(min_nanos / 1_000_000_000)
        dataset_end = datetime.utcfromtimestamp(max_nanos / 1_000_000_000)

        overlap_start = max(dataset_start, start_dt)
        overlap_end = min(dataset_end, end_dt)
        overlap_seconds = max((overlap_end - overlap_start).total_seconds(), 0.0)
        coverage = (overlap_seconds / required_seconds * 100.0) if required_seconds > 0 else 100.0

    summary = pl.DataFrame(
        {
            "expected_start": [start_dt],
            "expected_end": [end_dt],
            "dataset_start": [dataset_start],
            "dataset_end": [dataset_end],
            "overlap_duration_seconds": [overlap_seconds],
            "required_duration_seconds": [required_seconds],
            "temporal_coverage": [coverage],
            "threshold": [threshold],
            "status": ["pass" if coverage >= threshold else "fail"],
        }
    ).lazy()

    return df, {"temporal_coverage": summary}


def _parse_datetime(value: datetime | str, name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a datetime or ISO 8601 string") from exc
    raise TypeError(f"{name} must be a datetime or ISO 8601 string")
