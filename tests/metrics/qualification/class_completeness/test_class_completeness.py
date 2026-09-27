"""."""

import betterosi
from omega_prime.recording import Recording
import pytest
import polars as pl

from omega_prime.metrics.qualification.class_completeness import (
    TYPE_COMPLETENESS,
    SUBTYPE_COMPLETENESS,
    ROLE_COMPLETENESS,
    class_completeness,
    type_completeness,
)

from .conftest import assert_class_completeness

expected_pass = [
    betterosi.MovingObject.Type.PEDESTRIAN,
    betterosi.MovingObject.Type.VEHICLE,
]

expected_fail = [
    betterosi.MovingObject.Type.PEDESTRIAN,
    betterosi.MovingObject.Type.VEHICLE,
    betterosi.MovingObject.Type.ANIMAL,
]
vct = betterosi.MovingObject.VehicleClassification.Type
vcr = betterosi.MovingObject.VehicleClassification.Role


@pytest.fixture()
def class_df() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "type": [
                int(betterosi.MovingObject.Type.PEDESTRIAN),
                int(betterosi.MovingObject.Type.VEHICLE),
                int(betterosi.MovingObject.Type.VEHICLE),
            ]
        }
    ).lazy()


@pytest.fixture()
def class_df_with_subtype() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "type": [
                int(betterosi.MovingObject.Type.PEDESTRIAN),
                int(betterosi.MovingObject.Type.VEHICLE),
                int(betterosi.MovingObject.Type.VEHICLE),
            ],
            "subtype": [
                -1,
                int(vct.CAR),
                int(vct.BICYCLE),
            ],
        }
    ).lazy()


@pytest.fixture()
def class_df_with_role() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "type": [
                int(betterosi.MovingObject.Type.PEDESTRIAN),
                int(betterosi.MovingObject.Type.VEHICLE),
                int(betterosi.MovingObject.Type.VEHICLE),
            ],
            "role": [
                -1,
                int(vcr.CIVIL),
                int(vcr.POLICE),
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
        expected_subtypes=[vct.BICYCLE],
        expected_roles=[vcr.CIVIL],
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
    expected_subtypes = [vct.CAR, vct.BICYCLE]
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
    expected_subtypes = [vct.CAR, vct.BICYCLE, vct.BUS]
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
    expected_roles = [vcr.CIVIL, vcr.POLICE]
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
    expected_roles = [vcr.CIVIL, vcr.POLICE, vcr.AMBULANCE]
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


def test_vehicle_specific_expectations_are_ignored_without_vehicle_type(class_df) -> None:
    _df, result_dict = class_completeness(
        class_df,
        expected_types=[betterosi.MovingObject.Type.PEDESTRIAN],
        expected_subtypes=[vct.BUS],
        expected_roles=[vcr.AMBULANCE],
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


def test_type_completeness_requires_expected_types(class_df) -> None:
    with pytest.raises(ValueError, match="expected_types must be provided"):
        type_completeness(class_df, [])
