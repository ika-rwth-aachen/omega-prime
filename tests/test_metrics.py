import polars as pl
from omega_prime.metrics import ttc_and_thw


def test_ttc_and_thw_synthetic():
    # 'crossed' contains the continuous state of the ego and object over time.
    # In this scenario, at total_nanos=0, ego is at 0m driving at 10m/s.
    # Object is at 20m driving at 5m/s.
    crossed = pl.DataFrame(
        {
            "idx_ego": [1],
            "idx": [2],
            "total_nanos_ego": [0],
            "total_nanos": [0],
            "distance_traveled_ego": [0.0],
            "distance_traveled": [20.0],
            "vel_ego": [10.0],
            "vel": [5.0],
        }
    )

    # 'timegaps' acts as the point where the ego and object trajectories overlap
    # (i.e. the crossing point). In this scenario, they cross at the 100m mark.
    timegaps = pl.DataFrame(
        {"idx_ego": [1], "idx": [2], "distance_traveled_ego": [100.0], "distance_traveled": [100.0]}
    )

    # Expected TTC: object is 20m ahead, relative velocity is 5m/s -> 4.0 seconds.
    # Expected THW: object is 20m ahead, ego velocity is 10m/s -> 2.0 seconds.

    df_out, result = ttc_and_thw(None, ego_id=1, crossed=crossed, timegaps=timegaps)
    ttc_df = result["ttc_and_thw"]

    assert ttc_df.height == 1
    assert ttc_df["TTC"][0] == 4.0
    assert ttc_df["THW"][0] == 2.0


def test_ttc_and_thw_synthetic_no_overlap():
    # If the vehicle is in front but driving faster, TTC should be null
    crossed = pl.DataFrame(
        {
            "idx_ego": [1],
            "idx": [2],
            "total_nanos_ego": [0],
            "total_nanos": [0],
            "distance_traveled_ego": [0.0],
            "distance_traveled": [20.0],
            "vel_ego": [10.0],
            "vel": [15.0],
        }
    )

    timegaps = pl.DataFrame(
        {"idx_ego": [1], "idx": [2], "distance_traveled_ego": [100.0], "distance_traveled": [100.0]}
    )

    df_out, result = ttc_and_thw(None, ego_id=1, crossed=crossed, timegaps=timegaps)
    ttc_df = result["ttc_and_thw"]

    assert ttc_df.height == 1
    assert ttc_df["TTC"][0] is None
    assert ttc_df["THW"][0] == 2.0  # THW is independent of relative speed
