"""."""

from typing import Any
from collections.abc import Callable
import betterosi
from omega_prime.metrics.qualification.cli.vehicle_type import VehicleTypeCli


def test_osi_equivalence(from_osi: Callable[[list[Any]], list[str]]) -> None:
    osi_list = [v.name for v in betterosi.MovingObject.VehicleClassification.Type.desc().values]
    assert len(VehicleTypeCli.CHOICES) == len(osi_list)
    assert list(VehicleTypeCli.CHOICES) == from_osi(osi_list)
