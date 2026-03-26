"""."""

from omega_prime.metrics.qualification.cli.vehicle_role import VehicleRole


def test_get_default() -> None:
    assert VehicleRole.get_default() == tuple()
