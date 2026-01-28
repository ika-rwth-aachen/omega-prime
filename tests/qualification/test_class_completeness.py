"""."""

import betterosi
import pytest
import polars as pl

from omega_prime.qualification.class_completeness import class_completeness

@pytest.fixture()
def class_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "type": [
                int(betterosi.MovingObjectType.TYPE_PEDESTRIAN),
                int(betterosi.MovingObjectType.TYPE_VEHICLE),
                int(betterosi.MovingObjectType.TYPE_VEHICLE),
            ]
        }
    )

def test_pass(class_df) -> None:
    expected = [
        betterosi.MovingObjectType.TYPE_PEDESTRIAN,
        betterosi.MovingObjectType.TYPE_VEHICLE,
    ]
    _df, result_dict = class_completeness(class_df, expected_classes=expected)
    summary = result_dict["class_completeness"].collect()
    assert summary["coverage"][0] == pytest.approx(100.0)
    assert summary["status"][0] == "pass"


def test_fail(class_df) -> None:
    expected = [
        betterosi.MovingObjectType.TYPE_PEDESTRIAN,
        betterosi.MovingObjectType.TYPE_VEHICLE,
        betterosi.MovingObjectType.TYPE_ANIMAL,
    ]
    _df, result_dict = class_completeness(class_df, expected_classes=expected)
    summary = result_dict["class_completeness"].collect()
    assert summary["coverage"][0] == pytest.approx(66.66666666666667)
    assert summary["status"][0] == "fail"
