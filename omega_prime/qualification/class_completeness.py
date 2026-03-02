"""."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import betterosi
import polars as pl

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT

CLASS_COMPLETENESS = "class_completeness"
TYPE_COMPLETENESS = "moving_object_type_completeness"
SUBTYPE_COMPLETENESS = "vehicle_subtype_completeness"
ROLE_COMPLETENESS = "vehicle_role_completeness"


@metric(computes_properties=[CLASS_COMPLETENESS])
def class_completeness(
    df: pl.LazyFrame,
    /,
    expected_types: Sequence[betterosi.MovingObjectType],
    expected_subtypes: Sequence[betterosi.MovingObjectVehicleClassificationType] = tuple(),
    expected_roles: Sequence[betterosi.MovingObjectVehicleClassificationRole] = tuple(),
) -> QRT:
    type_completeness_score = type_completeness(df, expected_types)

    subtype_completeness_score = 100.0
    subtype_completeness_is_applicable = betterosi.MovingObjectType.TYPE_VEHICLE in expected_types
    if subtype_completeness_is_applicable:
        subtype_completeness_score = subtype_completeness(df, expected_subtypes)

    role_completeness_score = 100.0
    role_completeness_is_applicable = betterosi.MovingObjectType.TYPE_VEHICLE in expected_types
    if role_completeness_is_applicable:
        role_completeness_score = role_completeness(df, expected_roles)

    passes = (
        type_completeness_score > 99.999999999
        and (not subtype_completeness_is_applicable or subtype_completeness_score > 99.999999999)
        and (not role_completeness_is_applicable or role_completeness_score > 99.999999999)
    )
    status = PASS if passes else FAIL

    summary = pl.DataFrame(
        {
            CLASS_COMPLETENESS: min(type_completeness_score, subtype_completeness_score, role_completeness_score),
            TYPE_COMPLETENESS: type_completeness_score,
            SUBTYPE_COMPLETENESS: subtype_completeness_score if subtype_completeness_is_applicable else 100.0,
            ROLE_COMPLETENESS: role_completeness_score if role_completeness_is_applicable else 100.0,
            STATUS: [status],
        }
    ).lazy()

    return df, {CLASS_COMPLETENESS: summary}


def type_completeness(
    df: pl.LazyFrame,
    expected_types: Sequence[betterosi.MovingObjectType],
) -> float:
    if not expected_types:
        raise ValueError("expected_types must be provided")

    return _completeness_for_column(df, "type", expected_types, "expected_types")


def subtype_completeness(
    df: pl.LazyFrame,
    expected_subtypes: Sequence[betterosi.MovingObjectVehicleClassificationType] = tuple(),
) -> float:
    if not expected_subtypes:
        return 100.0
    return _completeness_for_column(df, "subtype", expected_subtypes, "expected_subtypes")


def role_completeness(
    df: pl.LazyFrame,
    expected_roles: Sequence[betterosi.MovingObjectVehicleClassificationRole] = tuple(),
) -> float:
    if not expected_roles:
        return 100.0
    return _completeness_for_column(df, "role", expected_roles, "expected_roles")


def _completeness_for_column(
    df: pl.LazyFrame,
    column: str,
    expected_values: Sequence[
        betterosi.MovingObjectType
        | betterosi.MovingObjectVehicleClassificationType
        | betterosi.MovingObjectVehicleClassificationRole
    ],
    expected_label: str,
) -> float:
    expected_set = _normalize_expected(expected_values)
    if not expected_set:
        raise ValueError(f"{expected_label} must contain at least one valid entry")

    try:
        observed = df.select(pl.col(column).drop_nulls().unique()).collect().get_column(column).to_list()
    except pl.exceptions.ColumnNotFoundError:
        observed = []
    observed_set = {v for v in observed if not (isinstance(v, int | float) and v < 0)}
    return (len(expected_set & observed_set) / len(expected_set)) * 100.0


def _normalize_expected(
    values: Sequence[
        betterosi.MovingObjectType
        | betterosi.MovingObjectVehicleClassificationType
        | betterosi.MovingObjectVehicleClassificationRole
    ],
) -> set[int]:
    expected: set[int] = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, Enum):
            v = int(v)
        if isinstance(v, int | float) and v < 0:
            continue
        expected.add(v)
    return expected
