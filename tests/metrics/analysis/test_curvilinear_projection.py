import numpy as np
import polars as pl
import shapely

from omega_prime.locator import ShapelyTrajectoryTools
from omega_prime.metrics import curvilinear_projection


def test_curvilinear_projection_synthetic():
    # Ego moves along the x-axis, straight line
    # Object moves along the y-axis, crossing the ego at x=50
    df = pl.DataFrame(
        {
            "idx": [1, 1, 2, 2],
            "total_nanos": [0, 10, 0, 10],
            "x": [0.0, 100.0, 50.0, 50.0],
            "y": [0.0, 0.0, -10.0, 10.0],
            # yaw = 0 for ego (moving along X), yaw = pi/2 for object (moving along Y)
            "yaw": [0.0, 0.0, np.pi / 2, np.pi / 2],
            "vel": [10.0, 10.0, 5.0, 5.0],
        }
    )

    df_out, result = curvilinear_projection(df.lazy(), ego_id=1)
    res = df_out.collect()

    # Check object (idx=2) projections
    obj_res = res.filter(pl.col("idx") == 2)

    # pos_lon for object should be 150.0 because it crosses ego at x=50, and ego line is extended by 100m backward
    assert np.isclose(obj_res["pos_lon"][0], 150.0)

    # curv_heading should be -pi/2 or pi/2 (object yaw is pi/2, reference heading is 0)
    assert np.isclose(np.abs(obj_res["curv_heading"][0]), np.pi / 2)

    # vel_lon should be roughly 0 since it is crossing perpendicularly
    assert np.isclose(obj_res["vel_lon"][0], 0.0, atol=1e-7)


def test_curvilinear_projection_nan_coordinates():
    # DataFrame with some NaN/invalid rows
    df = pl.DataFrame(
        {
            "idx": [1, 1, 2, 2],
            "total_nanos": [0, 10, 0, 10],
            "x": [0.0, 100.0, np.nan, 50.0],
            "y": [0.0, 0.0, np.nan, 10.0],
            "yaw": [0.0, 0.0, np.pi / 2, np.pi / 2],
            "vel": [10.0, 10.0, 5.0, 5.0],
        }
    )

    df_out, _ = curvilinear_projection(df.lazy(), ego_id=1)
    res = df_out.collect()

    obj_res = res.filter(pl.col("idx") == 2)
    # The first row for object (NaN inputs) should result in NaNs
    assert np.isnan(obj_res["pos_lon"][0])
    assert np.isnan(obj_res["curv_heading"][0])
    assert np.isnan(obj_res["vel_lon"][0])

    # The second row for object (valid inputs) should be computed correctly
    assert np.isclose(obj_res["pos_lon"][1], 150.0)
    assert np.isclose(np.abs(obj_res["curv_heading"][1]), np.pi / 2)
    assert np.isclose(obj_res["vel_lon"][1], 0.0, atol=1e-7)


def test_curvilinear_projection_all_invalid():
    # DataFrame where object has only NaN coordinates
    df = pl.DataFrame(
        {
            "idx": [1, 1, 2, 2],
            "total_nanos": [0, 10, 0, 10],
            "x": [0.0, 100.0, np.nan, np.nan],
            "y": [0.0, 0.0, np.nan, np.nan],
            "yaw": [0.0, 0.0, 0.0, 0.0],
            "vel": [10.0, 10.0, 5.0, 5.0],
        }
    )

    df_out, _ = curvilinear_projection(df.lazy(), ego_id=1)
    res = df_out.collect()

    obj_res = res.filter(pl.col("idx") == 2)
    assert np.all(np.isnan(obj_res["pos_lon"].to_numpy()))
    assert np.all(np.isnan(obj_res["curv_heading"].to_numpy()))
    assert np.all(np.isnan(obj_res["vel_lon"].to_numpy()))


def test_extend_linestring_of_a_standing_ego():
    # all positions coincide: the path has no length and no direction
    line = shapely.LineString([(5.0, 3.0), (5.0, 3.0), (5.0, 3.0)])
    with np.errstate(all="raise"):
        extended = ShapelyTrajectoryTools.extend_linestring(line, l_append=100, heading=np.pi / 4)
    assert np.isclose(extended.length, 200.0)
    # the position stays at s = l_append, the origin the callers subtract
    assert np.isclose(extended.project(shapely.Point(5.0, 3.0)), 100.0)
    # the axis runs along the heading
    start, end = np.asarray(extended.coords)[[0, -1]]
    assert np.allclose((end - start) / np.linalg.norm(end - start), [np.cos(np.pi / 4), np.sin(np.pi / 4)])


def test_extend_linestring_of_a_jittering_standing_ego():
    # start and end coincide and the jitter in between is below float precision at UTM coordinates
    line = shapely.LineString([(3e5, 5e6), (3e5 + 1e-9, 5e6), (3e5, 5e6)])
    with np.errstate(all="raise"):
        extended = ShapelyTrajectoryTools.extend_linestring(line, l_append=100)
    assert np.isclose(extended.length, 200.0)
    assert np.isclose(extended.project(shapely.Point(3e5, 5e6)), 100.0)


def standing_ego_df(ego_x, ego_y, ego_nanos):
    # standing ego (idx 1) facing along the x-axis, object (idx 2) driving ahead of it along the x-axis
    return pl.DataFrame(
        {
            "idx": [1] * len(ego_x) + [2, 2],
            "total_nanos": ego_nanos + [0, 10],
            "x": ego_x + [10.0, 20.0],
            "y": ego_y + [0.0, 0.0],
            "yaw": [0.0] * (len(ego_x) + 2),
            "vel": [0.0] * len(ego_x) + [5.0, 5.0],
        }
    )


def assert_object_ahead_of_standing_ego(df):
    with np.errstate(all="raise"):
        df_out, _ = curvilinear_projection(df.lazy(), ego_id=1)
        res = df_out.collect()
    ego_res = res.filter((pl.col("idx") == 1) & pl.col("x").is_not_nan())
    assert np.allclose(ego_res["pos_lon"].to_numpy(), 100.0)
    obj_res = res.filter(pl.col("idx") == 2)
    assert np.allclose(obj_res["pos_lon"].to_numpy(), [110.0, 120.0])
    assert np.allclose(obj_res["curv_heading"].to_numpy(), 0.0)
    assert np.allclose(obj_res["vel_lon"].to_numpy(), 5.0)


def test_curvilinear_projection_standing_ego():
    assert_object_ahead_of_standing_ego(standing_ego_df([0.0, 0.0], [0.0, 0.0], [0, 10]))


def test_curvilinear_projection_standing_ego_with_dropout():
    assert_object_ahead_of_standing_ego(standing_ego_df([0.0, np.nan, 0.0], [0.0, np.nan, 0.0], [0, 5, 10]))


def test_curvilinear_projection_ego_with_a_single_position():
    assert_object_ahead_of_standing_ego(standing_ego_df([0.0], [0.0], [0]))


def test_curvilinear_projection_no_valid_ego_position():
    df = standing_ego_df([np.nan, np.nan], [np.nan, np.nan], [0, 10])
    df_out, _ = curvilinear_projection(df.lazy(), ego_id=1)
    res = df_out.collect()
    assert np.all(np.isnan(res["pos_lon"].to_numpy()))
    assert np.all(np.isnan(res["vel_lon"].to_numpy()))
