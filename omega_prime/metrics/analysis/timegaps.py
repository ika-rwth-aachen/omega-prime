@metric(
    requires_columns=["distance_traveled", "vel"],
    computes_properties=["timegaps", "min_timegaps"],
    computes_intermediate_properties=["crossed"],
)
def timegaps_and_min_timgaps(df, /, ego_id, time_buffer=2e9):
    """Metrics that computes timegaps between `ego_id` and all other objects. `time_buffer` gives the timespan in which intersection of trajectories is tested"""
    ego_df = df.filter(idx=ego_id)

    crossed = df.join(ego_df, how="cross", suffix="_ego")

    crossed = crossed.filter(
        (pl.col("total_nanos_ego") - time_buffer) <= pl.col("total_nanos"),
        (pl.col("total_nanos_ego") + time_buffer) >= pl.col("total_nanos"),
        pl.col("idx_ego") != pl.col("idx"),
    )

    all_timegaps = (
        crossed.filter(pl.col("geometry").st.intersects(pl.col("geometry_ego")))
        .with_columns(timegap=(pl.col("total_nanos") - pl.col("total_nanos_ego")) / 1e9)
        .select(
            "idx_ego", "idx", "total_nanos_ego", "total_nanos", "timegap", "distance_traveled", "distance_traveled_ego"
        )
    )

    timegaps = (
        all_timegaps.group_by("idx", "idx_ego", "total_nanos_ego")
        .agg(
            pl.col("timegap", "total_nanos", "distance_traveled", "distance_traveled_ego").get(
                pl.col("timegap").abs().arg_min()
            ),
        )
        .sort("idx_ego", "idx", "total_nanos_ego")
        .select(
            "idx_ego", "idx", "total_nanos_ego", "timegap", "total_nanos", "distance_traveled", "distance_traveled_ego"
        )
    )
    min_timegaps = timegaps.group_by("idx_ego", "idx").agg(
        pl.col("timegap").get(pl.col("timegap").abs().arg_min()).alias("min_timegap")
    )

    return df, {"timegaps": timegaps, "min_timegaps": min_timegaps, "crossed": crossed}