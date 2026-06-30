import pytest
import polars as pl
from omega_prime.metrics import ttc_and_thw


@pytest.mark.filterwarnings("ignore:Sortedness of columns cannot be checked:UserWarning")
def test_ttc_and_thw_synthetic():
    # 'df' contains the main trajectory info (with curvilinear projections)
    df = pl.DataFrame(
        {
            "idx": [1, 2],
            "total_nanos": [0, 0],
            "pos_lon": [0.0, 20.0],
            "vel_lon": [10.0, 5.0],
            "distance_traveled": [0.0, 20.0],
            "vel": [10.0, 5.0],
        }
    )

    crossed = pl.DataFrame()  # Not used by the new ttc_and_thw logic!

    # 'timegaps' acts as the point where the ego and object trajectories overlap
    timegaps = pl.DataFrame({"idx_ego": [1], "idx": [2], "total_nanos_ego": [0], "total_nanos": [0]})

    # Expected TTC: object is 20m ahead, relative velocity is 5m/s -> 4.0 seconds.
    # Expected THW: object is 20m ahead, ego velocity is 10m/s -> 2.0 seconds.

    df_out, result = ttc_and_thw(df.lazy(), ego_id=1, crossed=crossed, timegaps=timegaps.lazy())
    ttc_df = result["ttc_and_thw"].collect()


    assert ttc_df.height == 1
    assert ttc_df["TTC"][0] == pytest.approx(4.0)
    assert ttc_df["THW"][0] == pytest.approx(2.0)


@pytest.mark.filterwarnings("ignore:Sortedness of columns cannot be checked:UserWarning")
def test_ttc_and_thw_synthetic_no_overlap():
    df = pl.DataFrame(
        {
            "idx": [1, 2],
            "total_nanos": [0, 0],
            "pos_lon": [0.0, 20.0],
            "vel_lon": [10.0, 15.0],
            "distance_traveled": [0.0, 20.0],
            "vel": [10.0, 15.0],
        }
    )

    crossed = pl.DataFrame()

    timegaps = pl.DataFrame({"idx_ego": [1], "idx": [2], "total_nanos_ego": [0], "total_nanos": [0]})

    df_out, result = ttc_and_thw(df.lazy(), ego_id=1, crossed=crossed, timegaps=timegaps.lazy())
    ttc_df = result["ttc_and_thw"].collect()


    assert ttc_df.height == 1
    assert ttc_df["TTC"][0] is None
    assert ttc_df["THW"][0] == pytest.approx(2.0)  # THW is independent of relative speed


def test_curvilinear_projection_synthetic():
    from omega_prime.metrics import curvilinear_projection
    import numpy as np

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
