"""."""

from omega_prime.metrics.qualification.cli.common import to_osi_str


def test_to_osi_str() -> None:
    assert to_osi_str("test") == "TEST"
    assert to_osi_str("test-a") == "TEST_A"
    assert to_osi_str("test-1-2") == "TEST_1_2"
