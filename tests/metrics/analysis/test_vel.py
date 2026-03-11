"""."""

import pytest

from omega_prime import Recording
from omega_prime.qualification.vel import vel


def test_vel(rec: Recording) -> None:
    lf, prop_dct = vel(rec.df)
    assert prop_dct == {}
    assert "vel" in lf.collect_schema()
    vv = lf.select("vel").collect().to_numpy()
    assert vv.shape == (868, 1)
    ref = [1.5000093131334005, 0.0, 1.5000092312622042, 0.0, 1.5000091532883408]
    assert vv[-5:, 0] == pytest.approx(ref)
