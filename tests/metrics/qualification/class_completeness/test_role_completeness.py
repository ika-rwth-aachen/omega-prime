"""."""

import betterosi
import pytest
import polars as pl

from omega_prime.metrics.qualification.class_completeness import role_completeness


vcr = betterosi.MovingObjectVehicleClassificationRole


@pytest.fixture()
def role_df() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "role": [
                int(vcr.ROLE_CIVIL),
                int(vcr.ROLE_POLICE),
                int(vcr.ROLE_CIVIL),
            ]
        }
    ).lazy()


def test_role_completeness_pass(role_df: pl.LazyFrame) -> None:
    expected_roles = [vcr.ROLE_CIVIL, vcr.ROLE_POLICE]
    result = role_completeness(role_df, expected_roles)
    assert result == pytest.approx(100.0)


def test_role_completeness_fail(role_df: pl.LazyFrame) -> None:
    expected_roles = [vcr.ROLE_CIVIL, vcr.ROLE_POLICE, vcr.ROLE_AMBULANCE]
    result = role_completeness(role_df, expected_roles)
    assert result == pytest.approx(66.66666666666667)
