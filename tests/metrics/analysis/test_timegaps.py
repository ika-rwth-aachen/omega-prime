"""."""

import numpy as np
import polars as pl
import polars_st  # noqa: F401
import pytest

from omega_prime import Recording

from omega_prime.metrics.analysis.timegaps import timegaps_and_min_timegaps
from omega_prime.metrics.analysis.distance_traveled import distance_traveled


def test_it(cut_in: Recording) -> None:
    with_distance, distance_dct = distance_traveled(cut_in.df)
    df_out, qualification_dct = timegaps_and_min_timegaps(with_distance, ego_id=0)
    assert "distance_traveled" in df_out.collect_schema()
    assert len(qualification_dct) == 3
    gaps = qualification_dct["timegaps"]
    assert len(gaps.collect_schema().names()) == 7
    min_gaps = qualification_dct["min_timegaps"].collect()
    assert min_gaps.to_numpy() == pytest.approx(np.array([[0, 1, -0.066]]))
    crossed = qualification_dct["crossed"].collect()
    assert crossed.height == 33245
    assert {"idx", "idx_ego", "total_nanos", "total_nanos_ego", "distance_traveled", "distance_traveled_ego"} <= set(
        crossed.columns
    )


def _timegaps_by_cross_join(df, ego_id, time_buffer):
    "the cross join `timegaps_and_min_timegaps` used to do: every row with every ego row, then the time window"
    crossed = df.join(df.filter(idx=ego_id), how="cross", suffix="_ego").filter(
        (pl.col("total_nanos_ego") - time_buffer) <= pl.col("total_nanos"),
        (pl.col("total_nanos_ego") + time_buffer) >= pl.col("total_nanos"),
        pl.col("idx_ego") != pl.col("idx"),
    )
    return (
        crossed.filter(pl.col("geometry").st.intersects(pl.col("geometry_ego")))
        .with_columns(timegap=(pl.col("total_nanos") - pl.col("total_nanos_ego")) / 1e9)
        .group_by("idx", "idx_ego", "total_nanos_ego")
        .agg(
            pl.col("timegap", "total_nanos", "distance_traveled", "distance_traveled_ego").get(
                pl.col("timegap").abs().arg_min()
            )
        )
        .sort("idx_ego", "idx", "total_nanos_ego")
        .select(
            "idx_ego", "idx", "total_nanos_ego", "timegap", "total_nanos", "distance_traveled", "distance_traveled_ego"
        )
    )


@pytest.mark.parametrize("time_buffer", [2e9, 10e9, 1_000_000_000])
def test_range_join_gives_the_cross_join_result(cut_in: Recording, time_buffer) -> None:
    with_distance, _ = distance_traveled(cut_in.df)
    _, dct = timegaps_and_min_timegaps(with_distance, ego_id=0, time_buffer=time_buffer)
    expected = _timegaps_by_cross_join(with_distance, 0, time_buffer).collect()
    assert expected.height > 0
    assert dct["timegaps"].collect().equals(expected)
