"""."""

import numpy as np
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
