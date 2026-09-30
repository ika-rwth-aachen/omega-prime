"""."""

from omega_prime.metrics.qualification.cli.vehicle_role import VehicleRoleCli
import betterosi

Role = betterosi.MovingObject.VehicleClassification.Role


def test_to_osi_list() -> None:
    assert VehicleRoleCli.to_osi_list([]) == []
    osi_list = VehicleRoleCli.to_osi_list(["police", "civil", "public-transport"])
    assert osi_list == [Role.POLICE, Role.CIVIL, Role.PUBLIC_TRANSPORT]
