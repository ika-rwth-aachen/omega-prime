"""."""

from omega_prime.metrics.qualification.cli.vehicle_role import VehicleRoleCli
from betterosi import MovingObjectVehicleClassificationRole as OsiRole


def test_to_osi_list() -> None:
    assert VehicleRoleCli.to_osi_list([]) == []
    osi_list = VehicleRoleCli.to_osi_list(["police", "civil", "public-transport"])
    assert osi_list == [OsiRole.POLICE, OsiRole.CIVIL, OsiRole.PUBLIC_TRANSPORT]
