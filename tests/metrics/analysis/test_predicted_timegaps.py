"""."""

import numpy as np
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
