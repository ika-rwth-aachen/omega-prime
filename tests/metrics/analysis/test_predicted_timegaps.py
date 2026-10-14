"""."""

import numpy as np
import polars as pl
import pytest

from omega_prime import Recording

from omega_prime.metrics.analysis.distance_traveled import distance_traveled
from omega_prime.metrics.analysis.predicted_timegaps import p_timegaps_and_min_p_timegaps
from omega_prime.metrics.analysis.timegaps import timegaps_and_min_timegaps


def test_it(cut_in: Recording) -> None:
    with_distance, distance_dct = distance_traveled(cut_in.df)
    with_timegaps, timegaps_dct = timegaps_and_min_timegaps(with_distance, ego_id=0)
    crossed = timegaps_dct["crossed"]
    timegaps = timegaps_dct["timegaps"]
    df_out, predicted_dct = p_timegaps_and_min_p_timegaps(with_timegaps, ego_id=0, crossed=crossed, timegaps=timegaps)
    assert "distance_traveled" in df_out.collect_schema()
    assert len(predicted_dct) == 2
    min_p_gaps = predicted_dct["min_p_timegaps"].collect()
    assert min_p_gaps.to_numpy() == pytest.approx(np.array([[0, 1, -9.26362707659445e-06]]))
    p_timegaps = predicted_dct["p_timegaps"].collect()
    assert p_timegaps.to_numpy().shape == (305, 5)


def _predicted(cut_in: Recording, overlap_horizon=None, time_buffer=2e9):
    with_distance, _ = distance_traveled(cut_in.df)
    with_timegaps, dct = timegaps_and_min_timegaps(with_distance, ego_id=0, time_buffer=time_buffer)
    _, predicted = p_timegaps_and_min_p_timegaps(
        with_timegaps,
        ego_id=0,
        crossed=dct["crossed"],
        timegaps=dct["timegaps"],
        time_buffer=time_buffer,
        overlap_horizon=overlap_horizon,
    )
    return predicted["p_timegaps"].collect(), dct["timegaps"].collect()


def test_horizon_longer_than_the_recording_changes_nothing(cut_in: Recording) -> None:
    unlimited, _ = _predicted(cut_in)
    limited, _ = _predicted(cut_in, overlap_horizon=1e12)
    keys = ["idx_ego", "idx", "total_nanos_ego"]
    assert limited.select(*keys, "p_timegap").equals(unlimited.select(*keys, "p_timegap"))


@pytest.mark.parametrize("overlap_horizon", [1e9, 1_000_000_000])
def test_horizon_only_uses_overlaps_close_in_time(cut_in: Recording, overlap_horizon) -> None:
    unlimited, timegaps = _predicted(cut_in)
    limited, _ = _predicted(cut_in, overlap_horizon=overlap_horizon)
    assert 0 < limited.height < unlimited.height
    # every ego timestamp left over has an overlap of the pair within the horizon
    near = limited.join(timegaps.select("idx_ego", "idx", t_overlap="total_nanos_ego"), on=["idx_ego", "idx"]).filter(
        (pl.col("total_nanos_ego") - pl.col("t_overlap")).abs() <= overlap_horizon
    )
    assert near.select("idx_ego", "idx", "total_nanos_ego").unique().height == limited.height
    # the smallest predicted timegap over fewer overlaps is never smaller
    both = limited.join(unlimited, on=["idx_ego", "idx", "total_nanos_ego"], suffix="_all")
    assert (both["p_timegap"].abs() >= both["p_timegap_all"].abs() - 1e-12).all()
