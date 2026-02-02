"""."""

import pytest

from omega_prime import Recording, MovingObjectType
from omega_prime.qualification.common import PASS, FAIL, STATUS
from omega_prime.qualification.object_type_coverage import object_type_coverage, OBJECT_TYPE_COVERAGE


def test_pass(rec: Recording) -> None:
    expect_types = MovingObjectType.PEDESTRIAN, MovingObjectType.VEHICLE
    _df, result_dict = object_type_coverage(rec.df, expected_types=expect_types)
    otc_df = result_dict[OBJECT_TYPE_COVERAGE].collect()
    assert otc_df[OBJECT_TYPE_COVERAGE][0] == pytest.approx(100.0)
    assert otc_df[STATUS][0] == PASS


def test_fail(rec: Recording) -> None:
    expect_types = MovingObjectType.PEDESTRIAN, MovingObjectType.VEHICLE, MovingObjectType.ANIMAL
    _df, result_dict = object_type_coverage(rec.df, expected_types=expect_types)
    otc_df = result_dict[OBJECT_TYPE_COVERAGE].collect()
    assert otc_df[OBJECT_TYPE_COVERAGE][0] == pytest.approx(66.6666666666666666)
    assert otc_df[STATUS][0] == FAIL
