"""."""

import pytest

from omega_prime import Recording
from omega_prime.qualification.distance_traveled import distance_traveled


def test_distance_traveled(rec: Recording) -> None:
    lf, prop_dct = distance_traveled(rec.df)
    assert prop_dct == {}
    assert "distance_traveled" in lf.collect_schema()
    vv = lf.select("distance_traveled").collect().to_numpy()
    assert vv.shape == (868, 1)
    ref = [67.91549710386548, 67.91549710386548, 67.96499740320694, 67.96499740320694, 68.01449769996489]
    assert vv[-5:, 0] == pytest.approx(ref)
