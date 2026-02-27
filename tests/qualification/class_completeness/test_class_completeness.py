"""."""

import betterosi
from omega_prime.recording import Recording
import pytest
import polars as pl

from omega_prime.qualification.class_completeness import (
    class_completeness,
    CLASS_COMPLETENESS,
    ROLE_COMPLETENESS,
    SUBTYPE_COMPLETENESS,
    TYPE_COMPLETENESS,
)

from ..conftest import qualification_assert

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
    _df, result_dict = class_completeness(class_df, expected_type=expected_pass)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=ROLE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True)


def test_fail(class_df) -> None:
    _df, result_dict = class_completeness(class_df, expected_type=expected_fail)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, False, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, False, sub_metric_name=ROLE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False)

def test_record_pass(rec: Recording) -> None:
    _df, result_dict = class_completeness(rec.df.lazy(), expected_type=expected_pass)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=ROLE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True)

def test_record_fail(rec: Recording) -> None:
    _df, result_dict = class_completeness(rec.df.lazy(), expected_type=expected_pass, expected_subtype=[vct.TYPE_BICYCLE], expected_role=[vcr.ROLE_CIVIL])
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, False, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 0.0, False, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 0.0, False, sub_metric_name=ROLE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 0.0, False)

def test_record_fail(rec: Recording) -> None:
    _df, result_dict = class_completeness(rec.df.lazy(), expected_type=expected_fail)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, False, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, False, sub_metric_name=ROLE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False)


def test_subtype_pass(class_df_with_subtype) -> None:
    expected_subtypes = [vct.TYPE_CAR, vct.TYPE_BICYCLE]
    _df, result_dict = class_completeness(
        class_df_with_subtype,
        expected_type=expected_pass,
        expected_subtype=expected_subtypes,
    )
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True)


def test_subtype_fail(class_df_with_subtype) -> None:
    expected_subtypes = [vct.TYPE_CAR, vct.TYPE_BICYCLE, vct.TYPE_BUS]
    _df, result_dict = class_completeness(
        class_df_with_subtype,
        expected_type=expected_pass,
        expected_subtype=expected_subtypes,
    )
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, False, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False)


def test_subtype_not_required(class_df) -> None:
    _df, result_dict = class_completeness(
        class_df,
        expected_type=expected_pass,
        expected_subtype=None,
    )
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True)


def test_role_pass(class_df_with_role) -> None:
    expected_roles = [vcr.ROLE_CIVIL, vcr.ROLE_POLICE]
    _df, result_dict = class_completeness(
        class_df_with_role,
        expected_type=expected_pass,
        expected_role=expected_roles,
    )
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=ROLE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True)


def test_role_fail(class_df_with_role) -> None:
    expected_roles = [vcr.ROLE_CIVIL, vcr.ROLE_POLICE, vcr.ROLE_AMBULANCE]
    _df, result_dict = class_completeness(
        class_df_with_role,
        expected_type=expected_pass,
        expected_role=expected_roles,
    )
    
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, False, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, False, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False, sub_metric_name=ROLE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 66.66666666666667, False)


def test_role_not_required(class_df) -> None:
    _df, result_dict = class_completeness(
        class_df,
        expected_type=expected_pass,
        expected_role=None,
    )
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=TYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=SUBTYPE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True, sub_metric_name=ROLE_COMPLETENESS)
    qualification_assert(result_dict, CLASS_COMPLETENESS, 100.0, True)
