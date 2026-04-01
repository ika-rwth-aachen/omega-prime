"""."""

from omega_prime.metrics.qualification.cli.vehicle_role import VehicleRoleCli


def test_get_default() -> None:
    assert VehicleRoleCli.get_default() == tuple()
