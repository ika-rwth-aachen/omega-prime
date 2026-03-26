"""."""

from omega_prime.metrics.qualification.cli.vehicle_type import VehicleType
from betterosi import MovingObjectVehicleClassificationType as OsiCls


def test_to_osi_list() -> None:
    assert VehicleType.to_osi_list([]) == []
    osi_list = VehicleType.to_osi_list(["bicycle", "bus", "car", "standup-scooter"])
    assert osi_list == [OsiCls.BICYCLE, OsiCls.BUS, OsiCls.CAR, OsiCls.STANDUP_SCOOTER]
