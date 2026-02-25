"""."""

import betterosi
import pytest
import polars as pl

from omega_prime.qualification.class_completeness import class_completeness, CLASS_COMPLETENESS

from .conftest import qualification_assert

@pytest.fixture()
def class_df() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "type": [
                int(betterosi.MovingObjectType.TYPE_PEDESTRIAN),
                int(betterosi.MovingObjectType.TYPE_VEHICLE),
                int(betterosi.MovingObjectType.TYPE_VEHICLE),
            ]
        }
    ).lazy()

def test_pass(class_df) -> None:
    expected = [
        betterosi.MovingObjectType.TYPE_PEDESTRIAN,
        betterosi.MovingObjectType.TYPE_VEHICLE,
    ]
    _df, result_dict = class_completeness(class_df, expected_classes=expected)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True)


def test_fail(class_df) -> None:
    expected = [
        betterosi.MovingObjectType.TYPE_PEDESTRIAN,
        betterosi.MovingObjectType.TYPE_VEHICLE,
        betterosi.MovingObjectType.TYPE_ANIMAL,
    ]
    _df, result_dict = class_completeness(class_df, expected_classes=expected)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False)
