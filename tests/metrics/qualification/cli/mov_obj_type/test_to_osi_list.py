"""."""

from omega_prime.metrics.qualification.cli.mov_obj_type import MovObjTypeCli
from betterosi import MovingObjectType as OsiType


def test_to_osi_list() -> None:
    assert MovObjTypeCli.to_osi_list([]) == []
    osi_list = MovObjTypeCli.to_osi_list(["animal", "vehicle", "pedestrian"])
    assert osi_list == [OsiType.ANIMAL, OsiType.VEHICLE, OsiType.PEDESTRIAN]
