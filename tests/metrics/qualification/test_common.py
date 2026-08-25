"""."""

from omega_prime import Recording
from omega_prime.metrics.qualification.common import get_num_rows


def test_get_num_rows(rec: Recording) -> None:
    assert get_num_rows(rec.df.lazy()) == 868
