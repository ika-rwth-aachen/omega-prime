from typing import Any
from collections.abc import Callable

from betterosi import MovingObjectType

from omega_prime.metrics.qualification.cli.mov_obj_type import MovObjTypeCli


def test_osi_equivalence(from_osi: Callable[[list[Any]], list[str]]) -> None:
    osi_list = list(MovingObjectType.betterproto_renamed_proto_names_to_value())
    assert len(MovObjTypeCli.CHOICES) == len(osi_list)
    assert list(MovObjTypeCli.CHOICES) == from_osi(osi_list)
