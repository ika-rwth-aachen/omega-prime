from typing import Any
from collections.abc import Callable

import betterosi

from omega_prime.metrics.qualification.cli.mov_obj_type import MovObjTypeCli


def test_osi_equivalence(from_osi: Callable[[list[Any]], list[str]]) -> None:
    osi_list = [v.name for v in betterosi.MovingObject.Type.desc().values]
    assert len(MovObjTypeCli.CHOICES) == len(osi_list)
    assert list(MovObjTypeCli.CHOICES) == from_osi(osi_list)
