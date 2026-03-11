"""."""

import betterosi
from omega_prime.recording import Recording
import pytest
import polars as pl

from omega_prime.qualification.class_completeness import (
    class_completeness,
    ROLE_COMPLETENESS,
    SUBTYPE_COMPLETENESS,
    TYPE_COMPLETENESS,
)

from .conftest import assert_class_completeness

expected_pass = [
    betterosi.MovingObjectType.TYPE_PEDESTRIAN,
    betterosi.MovingObjectType.TYPE_VEHICLE,
]

expected_fail = [
    betterosi.MovingObjectType.TYPE_PEDESTRIAN,
    betterosi.MovingObjectType.TYPE_VEHICLE,
    betterosi.MovingObjectType.TYPE_ANIMAL,
]
vct = betterosi.MovingObjectVehicleClassificationType
vcr = betterosi.MovingObjectVehicleClassificationRole


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


@pytest.fixture()
def class_df_with_subtype() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "type": [
                int(betterosi.MovingObjectType.TYPE_PEDESTRIAN),
                int(betterosi.MovingObjectType.TYPE_VEHICLE),
                int(betterosi.MovingObjectType.TYPE_VEHICLE),
            ],
            "subtype": [
                -1,
                int(vct.TYPE_CAR),
                int(vct.TYPE_BICYCLE),
            ],
        }
    ).lazy()


@pytest.fixture()
def class_df_with_role() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "type": [
                int(betterosi.MovingObjectType.TYPE_PEDESTRIAN),
                int(betterosi.MovingObjectType.TYPE_VEHICLE),
                int(betterosi.MovingObjectType.TYPE_VEHICLE),
            ],
            "role": [
                -1,
                int(vcr.ROLE_CIVIL),
                int(vcr.ROLE_POLICE),
            ],
        }
    ).lazy()


def test_pass(class_df) -> None:
    _df, result_dict = class_completeness(class_df, expected_types=expected_pass)
    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 100.0,
            SUBTYPE_COMPLETENESS: 100.0,
            ROLE_COMPLETENESS: 100.0,
        },
        is_pass=True,
    )


def test_fail(class_df) -> None:
    _df, result_dict = class_completeness(class_df, expected_types=expected_fail)
    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 66.66666666666667,
            SUBTYPE_COMPLETENESS: 100.0,
            ROLE_COMPLETENESS: 100.0,
        },
        is_pass=False,
    )


def test_record_pass(rec: Recording) -> None:
    _df, result_dict = class_completeness(rec.df.lazy(), expected_types=expected_pass)
    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 100.0,
            SUBTYPE_COMPLETENESS: 100.0,
            ROLE_COMPLETENESS: 100.0,
        },
        is_pass=True,
    )


def test_record_subtype_fail(rec: Recording) -> None:
    _df, result_dict = class_completeness(
        rec.df.lazy(),
        expected_types=expected_pass,
        expected_subtypes=[vct.TYPE_BICYCLE],
        expected_roles=[vcr.ROLE_CIVIL],
    )

    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 100.0,
            SUBTYPE_COMPLETENESS: 0.0,
            ROLE_COMPLETENESS: 0.0,
        },
        is_pass=False,
    )


def test_record_type_fail(rec: Recording) -> None:
    _df, result_dict = class_completeness(rec.df.lazy(), expected_types=expected_fail)

    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 66.66666666666667,
            SUBTYPE_COMPLETENESS: 100.0,
            ROLE_COMPLETENESS: 100.0,
        },
        is_pass=False,
    )


def test_subtype_pass(class_df_with_subtype) -> None:
    expected_subtypes = [vct.TYPE_CAR, vct.TYPE_BICYCLE]
    _df, result_dict = class_completeness(
        class_df_with_subtype, expected_types=expected_pass, expected_subtypes=expected_subtypes
    )

    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 100.0,
            SUBTYPE_COMPLETENESS: 100.0,
            ROLE_COMPLETENESS: 100.0,
        },
        is_pass=True,
    )


def test_subtype_fail(class_df_with_subtype) -> None:
    expected_subtypes = [vct.TYPE_CAR, vct.TYPE_BICYCLE, vct.TYPE_BUS]
    _df, result_dict = class_completeness(
        class_df_with_subtype,
        expected_types=expected_pass,
        expected_subtypes=expected_subtypes,
    )

    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 100.0,
            SUBTYPE_COMPLETENESS: 66.66666666666667,
            ROLE_COMPLETENESS: 100.0,
        },
        is_pass=False,
    )


def test_subtype_not_required(class_df) -> None:
    _df, result_dict = class_completeness(class_df, expected_types=expected_pass)

    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 100.0,
            SUBTYPE_COMPLETENESS: 100.0,
            ROLE_COMPLETENESS: 100.0,
        },
        is_pass=True,
    )


def test_role_pass(class_df_with_role) -> None:
    expected_roles = [vcr.ROLE_CIVIL, vcr.ROLE_POLICE]
    _df, result_dict = class_completeness(
        class_df_with_role,
        expected_types=expected_pass,
        expected_roles=expected_roles,
    )

    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 100.0,
            SUBTYPE_COMPLETENESS: 100.0,
            ROLE_COMPLETENESS: 100.0,
        },
        is_pass=True,
    )


def test_role_fail(class_df_with_role) -> None:
    expected_roles = [vcr.ROLE_CIVIL, vcr.ROLE_POLICE, vcr.ROLE_AMBULANCE]
    _df, result_dict = class_completeness(
        class_df_with_role,
        expected_types=expected_pass,
        expected_roles=expected_roles,
    )

    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 100.0,
            SUBTYPE_COMPLETENESS: 100.0,
            ROLE_COMPLETENESS: 66.66666666666667,
        },
        is_pass=False,
    )


def test_role_not_required(class_df) -> None:
    _df, result_dict = class_completeness(class_df, expected_types=expected_pass)
    assert_class_completeness(
        result_dict,
        expected_values={
            TYPE_COMPLETENESS: 100.0,
            SUBTYPE_COMPLETENESS: 100.0,
            ROLE_COMPLETENESS: 100.0,
        },
        is_pass=True,
    )
