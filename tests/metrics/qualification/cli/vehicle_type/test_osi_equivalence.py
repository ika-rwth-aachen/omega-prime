"""."""

from typing import Any
from collections.abc import Callable
from betterosi import MovingObjectVehicleClassificationType
from omega_prime.metrics.qualification.cli.vehicle_type import VehicleType


def test_osi_equivalence(from_osi: Callable[[list[Any]], list[str]]) -> None:
    osi_list = list(MovingObjectVehicleClassificationType.betterproto_renamed_proto_names_to_value())
    assert len(VehicleType.CHOICES) == len(osi_list)
    assert list(VehicleType.CHOICES) == from_osi(osi_list)
