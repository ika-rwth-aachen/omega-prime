"""."""

from omega_prime import Recording
from omega_prime.qualification.common import get_num_rec


def test_get_num_rec(rec: Recording) -> None:
    assert get_num_rec(rec.df.lazy()) == 17360
