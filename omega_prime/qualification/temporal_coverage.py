"""."""

from datetime import UTC, datetime

import polars as pl

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT, get_num_rows


TEMPORAL_COVERAGE = "temporal_coverage"


@metric(computes_properties=[TEMPORAL_COVERAGE])
def temporal_coverage(
    df: pl.LazyFrame,
    /,
    expected_start: datetime | str,
    expected_end: datetime | str,
    threshold: float = 80.0,
) -> QRT:
    start_dt = _parse_datetime(expected_start, "expected_start")
    end_dt = _parse_datetime(expected_end, "expected_end")
    if end_dt <= start_dt:
        raise ValueError("expected_end must be after expected_start")

    required_seconds = (end_dt - start_dt).total_seconds()
    row_count = get_num_rows(df)
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
        dataset_start = datetime.fromtimestamp(min_nanos * 1e-9, UTC)
        dataset_end = datetime.fromtimestamp(max_nanos * 1e-9, UTC)

        overlap_start = max(dataset_start, start_dt)
        overlap_end = min(dataset_end, end_dt)
        overlap_seconds = max((overlap_end - overlap_start).total_seconds(), 0.0)
        coverage = (overlap_seconds / required_seconds * 100.0) if required_seconds > 0 else 100.0

    status = PASS if coverage >= threshold else FAIL

    summary = pl.DataFrame(
        {
            TEMPORAL_COVERAGE: [coverage],
            STATUS: [status],
        }
    ).lazy()

    return df, {TEMPORAL_COVERAGE: summary}


def _parse_datetime(value: datetime | str, name: str) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a datetime or ISO 8601 string") from exc
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    raise TypeError(f"{name} must be a datetime or ISO 8601 string")
