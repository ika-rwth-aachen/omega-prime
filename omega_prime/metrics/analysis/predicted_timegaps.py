@metric(
    requires_columns=["distance_traveled", "vel"],
    requires_properties=["crossed", "timegaps"],
    computes_properties=["p_timegaps", "min_p_timegaps"],
)
def p_timegaps_and_min_p_timgaps(df, /, ego_id, crossed, timegaps, time_buffer=2e9):
    """Metrics that computes a predicted timegap between `ego_id` and all other objects. `time_buffer` gives the timespan in which intersection of trajectories is tested. The prediction is based on constant velocity following the same trajectory as observed."""
    p_timegaps = (
        crossed.join(timegaps, how="right", suffix="_overlap", on=["idx", "idx_ego"])
        .with_columns(
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
        .agg(
            pl.col("p_timegap", "total_nanos")
            .sort_by(pl.col("p_timegap").abs(), descending=False, nulls_last=True)
            .first()
        )
        .sort("idx_ego", "idx", "total_nanos_ego")
    )

    min_p_timegaps = p_timegaps.group_by("idx_ego", "idx").agg(
        pl.col("p_timegap").sort_by(pl.col("p_timegap").abs(), descending=False).first()
    )

    return df, {
        "p_timegaps": p_timegaps,
        "min_p_timegaps": min_p_timegaps,
    }
