"""."""

import betterosi
import pytest
import polars as pl

from omega_prime.qualification.class_completeness import subtype_completeness


vct = betterosi.MovingObjectVehicleClassificationType


@pytest.fixture()
def subtype_df() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "subtype": [
                int(vct.TYPE_CAR),
                int(vct.TYPE_BICYCLE),
                int(vct.TYPE_CAR),
            ]
        }
    ).lazy()


def test_subtype_completeness_pass(subtype_df: pl.LazyFrame) -> None:
    expected_subtypes = [vct.TYPE_CAR, vct.TYPE_BICYCLE]
    result = subtype_completeness(subtype_df, expected_subtypes)
    assert result == pytest.approx(100.0)


def test_subtype_completeness_fail(subtype_df: pl.LazyFrame) -> None:
    expected_subtypes = [vct.TYPE_CAR, vct.TYPE_BICYCLE, vct.TYPE_BUS]
    result = subtype_completeness(subtype_df, expected_subtypes)
    assert result == pytest.approx(66.66666666666667)
