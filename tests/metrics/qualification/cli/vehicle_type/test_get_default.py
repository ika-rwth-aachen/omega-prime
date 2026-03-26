"""."""

from omega_prime.metrics.qualification.cli.vehicle_type import VehicleType


def test_get_default() -> None:
    assert VehicleType.get_default() == tuple()
