import polars as pl

from ..metric import metric


@metric(
    requires_columns=["vel_lon", "pos_lon", "distance_traveled", "vel"],
    requires_properties=["timegaps"],
    computes_properties=["ttc_and_thw"],
)
def ttc_and_thw(df, /, ego_id, timegaps):
    """Metric that computes TTC and THW between `ego_id` and all other objects.

    Longitudinal distance is the arc-length gap along the ego's curvilinear
    trajectory axis (obj.pos_lon - ego.pos_lon), matching the legacy
    implementation.

    Object velocity is projected onto that axis via vel_lon = cos(curv_heading)
    * vel, which is the correct longitudinal component. The legacy code
    used sin() (the lateral component) — that was a bug; this version fixes it.
    """
    # Pull pos_lon and vel_lon for objects and ego separately so we can form
    # the longitudinal gap and closing speed at each ego timestep.
    obj_curv = (
        df.filter(pl.col("idx") != ego_id)
        .select(["idx", "total_nanos", "pos_lon", "vel_lon"])
        .rename({"total_nanos": "total_nanos_obj"})
    )

    ego_curv = (
        df.filter(pl.col("idx") == ego_id)
        .select(["total_nanos", "pos_lon", "vel_lon"])
        .rename(
            {
                "total_nanos": "total_nanos_ego",
                "pos_lon": "pos_lon_ego",
                "vel_lon": "vel_lon_ego",
            }
        )
    )

    ttc_df = (
        # Start from timegaps (one row per ego-timestep / object pair at the
        # crossing point) — this gives us the reference total_nanos_ego.
        timegaps.sort(["idx", "total_nanos_ego"])
        # Attach object curvilinear state at the moment nearest total_nanos_ego.
        # We use an asof join on total_nanos so we get the object's pos_lon at
        # the ego timestamp even if the object's own timestamps don't align
        # perfectly.
        .join_asof(
            obj_curv.sort(["idx", "total_nanos_obj"]),
            left_on="total_nanos_ego",
            right_on="total_nanos_obj",
            by_left="idx",
            by_right="idx",
            strategy="nearest",
        )
        # Attach ego curvilinear state at total_nanos_ego.
        .join(ego_curv, on="total_nanos_ego", how="left")
        .with_columns(
            # Positive lon_dist: object is ahead of ego on the reference line.
            lon_dist=(pl.col("pos_lon") - pl.col("pos_lon_ego")),
        )
        .with_columns(
            TTC=pl.when((pl.col("lon_dist") > 0) & (pl.col("vel_lon_ego") > pl.col("vel_lon")))
            .then(pl.col("lon_dist") / (pl.col("vel_lon_ego") - pl.col("vel_lon")))
            .otherwise(None),
            THW=pl.when((pl.col("lon_dist") > 0) & (pl.col("vel_lon_ego") > 0))
            .then(pl.col("lon_dist") / pl.col("vel_lon_ego"))
            .otherwise(None),
        )
        .group_by("idx_ego", "idx", "total_nanos_ego")
        .agg(
            pl.col("TTC", "THW", "total_nanos").sort_by(pl.col("TTC").abs(), descending=False, nulls_last=True).first()
        )
        .sort("idx_ego", "idx", "total_nanos_ego")
    )

    return df, {"ttc_and_thw": ttc_df}
