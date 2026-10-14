import polars as pl

from ..metric import MRT, metric

# ego timestamps joined at a time with `overlap_horizon`; the pairs of one block have to fit into memory
HORIZON_BLOCK = 2e9


def _predicted_timegaps(pairs: pl.LazyFrame, tie_breakers: list[pl.Expr]) -> pl.LazyFrame:
    "The smallest predicted timegap per ego timestamp, from pairs of crossed rows and overlaps (`timegaps` rows)."
    sort_key = [pl.col("p_timegap").abs(), *tie_breakers] if tie_breakers else pl.col("p_timegap").abs()
    return (
        pairs.with_columns(
            pl.when(pl.col("total_nanos") >= pl.col("total_nanos_overlap"))
            .then((pl.col("total_nanos_overlap") - pl.col("total_nanos")) / 1e9)
            .otherwise((pl.col("distance_traveled_overlap") - pl.col("distance_traveled")) / pl.col("vel"))
            .alias("time_to_overlap"),
            pl.when(pl.col("total_nanos_ego") >= pl.col("total_nanos_ego_overlap"))
            .then((pl.col("total_nanos_ego_overlap") - pl.col("total_nanos_ego")) / 1e9)
            .otherwise((pl.col("distance_traveled_ego_overlap") - pl.col("distance_traveled_ego")) / pl.col("vel_ego"))
            .alias("time_to_overlap_ego"),
        )
        .with_columns(
            -(
                pl.col("time_to_overlap_ego")
                - pl.col("time_to_overlap")
                + (pl.col("total_nanos_ego") - pl.col("total_nanos")) / 1e9
            ).alias("p_timegap")
        )
        .group_by("idx_ego", "idx", "total_nanos_ego")
        .agg(pl.col("p_timegap", "total_nanos").sort_by(sort_key, descending=False, nulls_last=True).first())
    )


def _within_horizon(df, ego_id, crossed, timegaps, overlap_horizon) -> list[pl.LazyFrame]:
    """
    The predicted timegaps for overlaps within `overlap_horizon` of the ego timestamp, one frame per block of
    `HORIZON_BLOCK` nanoseconds of ego timestamps.

    A join on the pair of objects alone combines every crossed row with every overlap of the pair before the horizon
    can be applied - for an object ahead in the ego's lane that is every timestamp. Per block, only the overlaps near
    it are joined, and each block is aggregated on its own.
    """
    first, last = (
        df.filter(idx=ego_id)
        .select(first=pl.col("total_nanos").min(), last=pl.col("total_nanos").max())
        .collect()
        .row(0)
    )
    if first is None:
        return []
    block, margin = int(HORIZON_BLOCK), int(overlap_horizon) + 1
    # compared in the type `total_nanos_ego - overlap_horizon` has, like the window in `timegaps`
    as_horizon_type = (lambda c: c) if isinstance(overlap_horizon, int) else (lambda c: c.cast(pl.Float64))
    nanos_ego, nanos_overlap = (
        as_horizon_type(pl.col("total_nanos_ego")),
        as_horizon_type(pl.col("total_nanos_ego_overlap")),
    )
    # collected once: every block reads them, and the lazy plan would compute them again for each
    crossed = (
        crossed.select(
            "idx",
            "idx_ego",
            "total_nanos",
            "total_nanos_ego",
            "distance_traveled",
            "distance_traveled_ego",
            "vel",
            "vel_ego",
        )
        .collect()
        .lazy()
    )
    timegaps = timegaps.collect().lazy()
    frames = []
    for start in range(int(first), int(last) + 1, block):
        end = start + block
        # a superset of the overlaps within the horizon of the block, narrowed exactly after the join
        overlaps = timegaps.filter(
            pl.col("total_nanos_ego") >= start - margin, pl.col("total_nanos_ego") <= end + margin
        )
        pairs = (
            crossed.filter(pl.col("total_nanos_ego") >= start, pl.col("total_nanos_ego") < end)
            .join(overlaps, on=["idx", "idx_ego"], how="inner", suffix="_overlap")
            .filter(nanos_ego >= nanos_overlap - overlap_horizon, nanos_ego <= nanos_overlap + overlap_horizon)
        )
        # the order of a join is not fixed, so equal predicted timegaps are decided by time
        frames.append(_predicted_timegaps(pairs, [pl.col("total_nanos"), pl.col("total_nanos_ego_overlap")]))
    return frames


@metric(
    requires_columns=["distance_traveled", "vel"],
    requires_properties=["crossed", "timegaps"],
    computes_properties=["p_timegaps", "min_p_timegaps"],
)
def p_timegaps_and_min_p_timegaps(
    df: pl.LazyFrame, /, ego_id, crossed, timegaps, time_buffer=2e9, overlap_horizon=None
) -> MRT:
    """Metrics that computes a predicted timegap between `ego_id` and all other objects. `time_buffer` gives the timespan in which intersection of trajectories is tested. The prediction is based on constant velocity following the same trajectory as observed.

    `overlap_horizon` [ns] limits the overlaps a predicted timegap is computed for to those whose ego timestamp lies
    within this horizon of the ego timestamp. Without it (`None`) every overlap of the pair counts; for an object
    that drives ahead in the ego's lane that is every timestamp, so the work grows with the cube of the track length.
    With it, an ego timestamp without an overlap inside the horizon gets no predicted timegap.
    """
    if overlap_horizon is None:
        p_timegaps = _predicted_timegaps(
            crossed.join(timegaps, how="right", suffix="_overlap", on=["idx", "idx_ego"]), []
        ).sort("idx_ego", "idx", "total_nanos_ego")
    else:
        frames = _within_horizon(df, ego_id, crossed, timegaps, overlap_horizon)
        if not frames:
            frames = [_predicted_timegaps(crossed.join(timegaps, suffix="_overlap", on=["idx", "idx_ego"]).clear(), [])]
        p_timegaps = pl.concat(frames, parallel=False).sort("idx_ego", "idx", "total_nanos_ego")

    min_p_timegaps = p_timegaps.group_by("idx_ego", "idx").agg(
        pl.col("p_timegap").sort_by(pl.col("p_timegap").abs(), descending=False).first()
    )

    return df, {
        "p_timegaps": p_timegaps,
        "min_p_timegaps": min_p_timegaps,
    }
