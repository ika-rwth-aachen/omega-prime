"""."""

import pytest

from omega_prime import Recording, MovingObjectType
from omega_prime.qualification.object_type_coverage import object_type_coverage


def test_pass(rec: Recording) -> None:
    expect_types = MovingObjectType.PEDESTRIAN, MovingObjectType.VEHICLE
    _df, result_dict = object_type_coverage(rec.df, expected_types=expect_types)
    otc_df = result_dict["object_type_coverage"].collect()
    assert otc_df["object_type_coverage"][0] == pytest.approx(100.0)
    assert otc_df["status"][0] == "pass"


def test_fail(rec: Recording) -> None:
    expect_types = MovingObjectType.PEDESTRIAN, MovingObjectType.VEHICLE, MovingObjectType.ANIMAL
    _df, result_dict = object_type_coverage(rec.df, expected_types=expect_types)
    otc_df = result_dict["object_type_coverage"].collect()
    assert otc_df["object_type_coverage"][0] == pytest.approx(66.6666666666666666)
    assert otc_df["status"][0] == "fail"
