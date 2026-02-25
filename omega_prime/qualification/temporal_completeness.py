"""."""

import polars as pl

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT, get_num_rows

TEMPORAL_COMPLETENESS = "temporal_completeness"


@metric(computes_properties=[TEMPORAL_COMPLETENESS])
def temporal_completeness(
    df: pl.LazyFrame,
    /,
    expected_frequency: float,
) -> QRT:
    if expected_frequency <= 0.0:
        raise ValueError(f"expected_frequency must be > 0, got {expected_frequency}")

    delta_target = 1.0 / expected_frequency
    track_threshold = 0.95

    idx_and_time = df.select("idx", "total_nanos").sort(["idx", "total_nanos"])
    deltas = idx_and_time.with_columns((pl.col("total_nanos").diff().over("idx") / 1e9).alias("delta_t"))
    valid_deltas = deltas.filter(pl.col("delta_t").is_not_null())
    per_track_rms = valid_deltas.group_by("idx").agg(
        (((pl.col("delta_t") - delta_target) / delta_target).pow(2).mean().sqrt()).alias("delta_rms")
    )
    below_count_query = per_track_rms.select((pl.col("delta_rms") > (1 - track_threshold)).sum().alias("below_count"))

    below_count = int(below_count_query.collect().item(0, 0))

    total_tracks = get_num_rows(per_track_rms)

    if total_tracks > 0:
        temporal_completeness = (1.0 - (below_count / total_tracks)) * 100.0
    else:
        temporal_completeness = 100.0

    summary = pl.DataFrame(
        {
            TEMPORAL_COMPLETENESS: temporal_completeness,
            STATUS: [PASS if temporal_completeness > 99.999999999 else FAIL],
        }
    ).lazy()

    return df, {TEMPORAL_COMPLETENESS: summary}
