"""."""

import math

import pytest

from omega_prime.qualification.target_area_coverage import _apply_proj_offset


def test_apply_proj_offset_pass() -> None:
    adjusted_coords = _apply_proj_offset(
        [(1.0, 0.0), (2.0, 0.0), (1.0, 1.0)],
        (1.0, 0.0, math.pi / 2),
    )

    assert adjusted_coords[0] == pytest.approx((0.0, 0.0))
    assert adjusted_coords[1] == pytest.approx((0.0, -1.0))
    assert adjusted_coords[2] == pytest.approx((1.0, 0.0))
