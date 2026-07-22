"""."""

import pytest
from typing import Any
from collections.abc import Callable


@pytest.fixture
def from_osi() -> Callable[[list[Any]], list[str]]:
    def osi_to_qualification(osi: list[Any]) -> list[str]:
        kk = sorted(osi)
        return ["-".join(k.split("_")[1:]).lower() for k in kk]

    return osi_to_qualification
