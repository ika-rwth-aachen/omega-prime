"""."""

import betterosi
import pytest
import polars as pl

from omega_prime.metrics.qualification.class_completeness import subtype_completeness


vct = betterosi.MovingObject.VehicleClassification.Type


@pytest.fixture()
def subtype_df() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "subtype": [
                int(vct.CAR),
                int(vct.BICYCLE),
                int(vct.CAR),
            ]
        }
    ).lazy()


def test_pass(subtype_df: pl.LazyFrame) -> None:
    expected_subtypes = [vct.CAR, vct.BICYCLE]
    result = subtype_completeness(subtype_df, expected_subtypes)
    assert result == pytest.approx(100.0)


def test_fail(subtype_df: pl.LazyFrame) -> None:
    expected_subtypes = [vct.CAR, vct.BICYCLE, vct.BUS]
    result = subtype_completeness(subtype_df, expected_subtypes)
    assert result == pytest.approx(66.66666666666667)


def test_missing_column_filters_invalid_expected_values() -> None:
    df_without_subtype = pl.DataFrame({"type": [0]}).lazy()

    result = subtype_completeness(df_without_subtype, [None, -1, vct.CAR])

    assert result == pytest.approx(0.0)


def test_only_invalid_expected_values(subtype_df: pl.LazyFrame) -> None:
    with pytest.raises(ValueError, match="expected_subtypes must contain at least one valid entry"):
        subtype_completeness(subtype_df, [None, -1])
