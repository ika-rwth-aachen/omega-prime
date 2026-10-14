import polars as pl
import polars_st  # noqa: F401

from ..metric import MRT, metric


@metric(
    requires_columns=["distance_traveled", "vel"],
    computes_properties=["timegaps", "min_timegaps"],
    computes_intermediate_properties=["crossed"],
)
def timegaps_and_min_timegaps(df: pl.LazyFrame, /, ego_id, time_buffer=2e9) -> MRT:
    """Metrics that computes timegaps between `ego_id` and all other objects. `time_buffer` gives the timespan in which intersection of trajectories is tested"""
    # Only the pairs of an object row and an ego row within `time_buffer` of each other, found by a range join: a
    # cross join of every row with every ego row does not fit into memory for a long ego track. The bounds are
    # compared in the type `total_nanos_ego - time_buffer <= total_nanos` compares them (floats for a float buffer),
    # and the rows are put back into the order of the cross join (object row, then ego row), which decides between
    # equal timegaps below.
    nanos = pl.col("total_nanos") if isinstance(time_buffer, int) else pl.col("total_nanos").cast(pl.Float64)
    rows = (
        df.select("idx", "total_nanos", "geometry", "distance_traveled", "vel")
        .with_row_index("_row")
        .with_columns(_nanos=nanos)
    )
    ego_rows = rows.filter(idx=ego_id).select(
        pl.all().exclude("_nanos"),
        _lower=pl.col("_nanos") - time_buffer,
        _upper=pl.col("_nanos") + time_buffer,
    )

    crossed = (
        rows.join_where(
            ego_rows,
            pl.col("_nanos") >= pl.col("_lower"),
            pl.col("_nanos") <= pl.col("_upper"),
            suffix="_ego",
        )
        .filter(pl.col("idx_ego") != pl.col("idx"))
        .sort("_row", "_row_ego")
        .drop("_row", "_row_ego", "_nanos", "_lower", "_upper")
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
