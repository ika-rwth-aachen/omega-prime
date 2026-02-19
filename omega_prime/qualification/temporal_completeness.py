"""."""

import polars as pl

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT

TEMPORAL_COMPLETENESS = "temporal_completeness"


@metric(computes_properties=[TEMPORAL_COMPLETENESS])
def temporal_completeness(
    df: pl.LazyFrame | pl.DataFrame,
    /,
    expected_frequency: float,
) -> QRT:
    DELTA_TARGET = 1.0 / expected_frequency
    TRACK_THRESHOLD = 0.95

    idx_and_time = df.select("idx", "total_nanos").sort(["idx", "total_nanos"])
    deltas = idx_and_time.with_columns((pl.col("total_nanos").diff().over("idx") / 1e9).alias("delta_t"))
    valid_deltas = deltas.filter(pl.col("delta_t").is_not_null())
    per_track_rms = valid_deltas.group_by("idx").agg(
        (((pl.col("delta_t") - DELTA_TARGET) / DELTA_TARGET).pow(2).mean().sqrt()).alias("delta_rms")
    )
    below_count_query = per_track_rms.select((pl.col("delta_rms") > (1 - TRACK_THRESHOLD)).sum().alias("below_count"))

    if isinstance(below_count_query, pl.LazyFrame):
        below_count = int(below_count_query.collect().item(0, 0))
    else:
        below_count = int(below_count_query.item(0, 0))

    if isinstance(per_track_rms, pl.LazyFrame):
        total_tracks = per_track_rms.select(pl.len()).collect().item(0, 0)
    else:
        total_tracks = per_track_rms.shape[0]
        
    if total_tracks > 0:
        temporal_completeness = (1.0 - (below_count / total_tracks)) * 100.0
    else:
        temporal_completeness = 100.0

    summary = pl.DataFrame(
        {
            TEMPORAL_COMPLETENESS: temporal_completeness,
            STATUS: [PASS if temporal_completeness == 100.0 else FAIL],
        }
    ).lazy()

    return df, {TEMPORAL_COMPLETENESS: summary}
