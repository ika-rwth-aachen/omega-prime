import numpy as np
import polars as pl
import shapely

from ...locator import ShapelyTrajectoryTools
from ..metric import metric


@metric(
    requires_columns=["vel"],
    computes_intermediate_columns=["pos_lon", "curv_heading", "vel_lon"],
)
def curvilinear_projection(df, /, ego_id) -> tuple[pl.LazyFrame, dict]:
    """Projects all objects onto the ego's curvilinear reference axis.

    Computes three intermediate columns:
      - pos_lon:      arc-length position along the ego's trajectory line
      - curv_heading: object heading relative to the reference tangent at its
                       projected position, in radians
      - vel_lon:      longitudinal velocity component (cos(curv_heading) * vel)

    These are intermediate-only columns: consumed by ttc_and_thw and then
    dropped from the final output by MetricManager.
    """
    # Collect only the ego rows to build the curvilinear reference line.
    # This is a small, targeted collect — the rest of the frame stays lazy.
    ego_xy = df.filter(pl.col("idx") == ego_id).sort("total_nanos").select(["x", "y"]).collect().to_numpy()

    ego_curvilinear = ShapelyTrajectoryTools.extend_linestring(shapely.LineString(ego_xy), l_append=100)

    # Collect only the scalar columns needed for Shapely operations, deliberately
    # excluding heavy geometry/polygon columns that are not used here.
    # Row order is preserved by the LazyFrame plan, so the resulting Series can
    # be attached back to `df` safely via with_columns.
    minimal = df.select(["x", "y", "yaw", "vel"]).collect()

    xy = minimal.select(["x", "y"]).to_numpy()
    points = shapely.points(xy[:, 0], xy[:, 1])

    # st[:, 0] = arc-length (pos_lon), st[:, 1] = lateral offset
    st_arr = ShapelyTrajectoryTools.xy2st(ego_curvilinear, points)
    pos_lon = st_arr[:, 0]

    # Tangent heading of the reference line at each projected point, in radians
    heading_ref_rad = ShapelyTrajectoryTools.st2xy(
        ego_curvilinear,
        st_arr[:, 0],
        st_arr[:, 1],
        return_heading_of_ref_at_st=True,
    )

    # yaw in the Recording dataframe is already in radians.
    yaw_rad = minimal["yaw"].to_numpy()
    # curv_heading: how much the object's heading deviates from the reference
    # tangent. 0 means perfectly aligned, ±pi/2 means perpendicular.
    curv_heading_rad = heading_ref_rad - yaw_rad

    # Longitudinal velocity: projection of speed onto the reference tangent.
    # cos(0) = 1 for aligned traffic, cos(pi/2) = 0 for crossing traffic.
    vel_arr = minimal["vel"].to_numpy()
    vel_lon = np.cos(curv_heading_rad) * vel_arr

    # Attach the computed arrays back to the original LazyFrame without
    # materialising its heavy columns (geometry/polygon stay lazy).
    return df.with_columns(
        [
            pl.Series("pos_lon", pos_lon, dtype=pl.Float64),
            pl.Series("curv_heading", curv_heading_rad, dtype=pl.Float64),
            pl.Series("vel_lon", vel_lon, dtype=pl.Float64),
        ]
    ), {}
