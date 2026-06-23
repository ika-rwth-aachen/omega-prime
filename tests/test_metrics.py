import polars as pl
from omega_prime.metrics import ttc_and_thw


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
    assert ttc_df["TTC"][0] == 4.0
    assert ttc_df["THW"][0] == 2.0


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
    assert ttc_df["THW"][0] == 2.0  # THW is independent of relative speed
