"""."""

import betterosi
import pytest
import polars as pl

from omega_prime.metrics.qualification.class_completeness import role_completeness


vcr = betterosi.MovingObject.VehicleClassification.Role


@pytest.fixture()
def role_df() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "role": [
                int(vcr.CIVIL),
                int(vcr.POLICE),
                int(vcr.CIVIL),
            ]
        }
    ).lazy()


def test_role_completeness_pass(role_df: pl.LazyFrame) -> None:
    expected_roles = [vcr.CIVIL, vcr.POLICE]
    result = role_completeness(role_df, expected_roles)
    assert result == pytest.approx(100.0)


def test_role_completeness_fail(role_df: pl.LazyFrame) -> None:
    expected_roles = [vcr.CIVIL, vcr.POLICE, vcr.AMBULANCE]
    result = role_completeness(role_df, expected_roles)
    assert result == pytest.approx(66.66666666666667)
