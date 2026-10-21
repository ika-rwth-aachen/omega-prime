"""."""

from omega_prime import Recording, MovingObjectType
from omega_prime.metrics.qualification.object_type_coverage import OBJECT_TYPE_COVERAGE, object_type_coverage

from .conftest import qualification_assert


def test_pass(rec: Recording) -> None:
    expect_types = MovingObjectType.PEDESTRIAN, MovingObjectType.VEHICLE
    _df, result_dict = object_type_coverage(rec.df, expected_types=expect_types)
    qualification_assert(result_dict, OBJECT_TYPE_COVERAGE, 100.0, True)


def test_fail(rec: Recording) -> None:
    expect_types = MovingObjectType.PEDESTRIAN, MovingObjectType.VEHICLE, MovingObjectType.ANIMAL
    _df, result_dict = object_type_coverage(rec.df, expected_types=expect_types)
    qualification_assert(result_dict, OBJECT_TYPE_COVERAGE, 66.6666666666666666, False)
