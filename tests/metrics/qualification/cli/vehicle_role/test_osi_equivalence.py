from typing import Any
from collections.abc import Callable

from betterosi import MovingObjectVehicleClassificationRole

from omega_prime.metrics.qualification.cli.vehicle_role import VehicleRoleCli


def test_osi_equivalence(from_osi: Callable[[list[Any]], list[str]]) -> None:
    osi_list = list(MovingObjectVehicleClassificationRole.betterproto_renamed_proto_names_to_value())
    assert len(VehicleRoleCli.CHOICES) == len(osi_list)
    assert list(VehicleRoleCli.CHOICES) == from_osi(osi_list)
