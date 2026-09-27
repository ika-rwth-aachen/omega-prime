from typing import Any
from collections.abc import Callable

import betterosi

Role = betterosi.MovingObject.VehicleClassification.Role

from omega_prime.metrics.qualification.cli.vehicle_role import VehicleRoleCli


def test_osi_equivalence(from_osi: Callable[[list[Any]], list[str]]) -> None:
    osi_list = [v.name for v in Role.desc().values]
    assert len(VehicleRoleCli.CHOICES) == len(osi_list)
    assert list(VehicleRoleCli.CHOICES) == from_osi(osi_list)
