"""."""

import betterosi
from omega_prime.recording import Recording
import pytest
import polars as pl

from omega_prime.qualification.class_completeness import class_completeness, CLASS_COMPLETENESS

from .conftest import qualification_assert

expected_pass = [
    betterosi.MovingObjectType.TYPE_PEDESTRIAN,
    betterosi.MovingObjectType.TYPE_VEHICLE,
]

expected_fail = [
    betterosi.MovingObjectType.TYPE_PEDESTRIAN,
    betterosi.MovingObjectType.TYPE_VEHICLE,
    betterosi.MovingObjectType.TYPE_ANIMAL,
]

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
    _df, result_dict = class_completeness(class_df, expected_classes=expected_pass)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True)

def test_fail(class_df) -> None:
    expected = expected_fail
    _df, result_dict = class_completeness(class_df, expected_classes=expected_fail)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False)

def test_record_pass(rec: Recording) -> None:
    df, result_dict = class_completeness(rec.df.lazy(), expected_classes=expected_pass)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True)

def test_record_fail(rec: Recording) -> None:
    df, result_dict = class_completeness(rec.df.lazy(), expected_classes=expected_fail)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False)