"""."""

import pytest

from omega_prime import Recording
from omega_prime.metrics.analysis.distance_traveled import distance_traveled


def test_distance_traveled(rec: Recording) -> None:
    lf, prop_dct = distance_traveled(rec.df)
    assert prop_dct == {}
    assert "distance_traveled" in lf.collect_schema()
    vv = lf.select("distance_traveled").collect().to_numpy()
    assert vv.shape == (868, 1)
    ref = [19.99649132018348, 47.919005783682024, 20.045991619524944, 47.919005783682024, 20.095491916282896]
    assert vv[-5:, 0] == pytest.approx(ref)
