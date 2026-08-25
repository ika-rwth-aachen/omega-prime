"""."""

from omega_prime.metrics.qualification.cli.vehicle_type import VehicleTypeCli


def test_get_default() -> None:
    assert VehicleTypeCli.get_default() == tuple()
