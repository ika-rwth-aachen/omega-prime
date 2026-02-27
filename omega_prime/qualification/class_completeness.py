"""."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import betterosi
import polars as pl

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT

CLASS_COMPLETENESS = "class_completeness"
SUBTYPE_COMPLETENESS = "vehicle_subtype_completeness"
ROLE_COMPLETENESS = "vehicle_role_completeness"


@metric(computes_properties=[CLASS_COMPLETENESS])
def class_completeness(
    df: pl.LazyFrame,
    /,
    expected_classes: Sequence[betterosi.MovingObjectType],
    expected_subtype: Sequence[betterosi.MovingObjectVehicleClassificationType] | None = None,
    expected_role: Sequence[betterosi.MovingObjectVehicleClassificationRole] | None = None,
) -> QRT:
    if not expected_classes:
        raise ValueError("expected_classes must be provided")

    expected_set = _normalize_expected(expected_classes)
    if not expected_set:
        raise ValueError("expected_classes must contain at least one valid entry")

    try:
        observed = df.select(pl.col("type").drop_nulls().unique()).collect().get_column("type").to_list()
    except pl.exceptions.ColumnNotFoundError:
        observed = []

    observed_set = {v for v in observed if not (isinstance(v, int | float) and v < 0)}
    class_completeness = (len(expected_set & observed_set) / len(expected_set)) * 100.0

    subtype_completeness_score = 100.0
    subtype_completeness_is_applicable = (
        expected_subtype is not None
        and len(expected_subtype) > 0
        and betterosi.MovingObjectType.TYPE_VEHICLE in expected_classes
    )
    if subtype_completeness_is_applicable:
        subtype_completeness_score = subtype_completeness(df, expected_subtype)

    role_completeness_score = 100.0
    role_completeness_is_applicable = (
        expected_role is not None
        and len(expected_role) > 0
        and betterosi.MovingObjectType.TYPE_VEHICLE in expected_classes
    )
    if role_completeness_is_applicable:
        role_completeness_score = role_completeness(df, expected_role)

    passes = (
        class_completeness > 99.999999999
        and (not subtype_completeness_is_applicable or subtype_completeness_score > 99.999999999)
        and (not role_completeness_is_applicable or role_completeness_score > 99.999999999)
    )
    status = PASS if passes else FAIL

    summary = pl.DataFrame(
        {
            CLASS_COMPLETENESS: class_completeness,
            SUBTYPE_COMPLETENESS: subtype_completeness_score if subtype_completeness_is_applicable else 100.0,
            ROLE_COMPLETENESS: role_completeness_score if role_completeness_is_applicable else 100.0,
            STATUS: [status],
        }
    ).lazy()

    return df, {CLASS_COMPLETENESS: summary}


def subtype_completeness(
    df: pl.LazyFrame,
    expected_subtype: Sequence[betterosi.MovingObjectVehicleClassificationType] | None,
) -> float:
    if not expected_subtype:
        return 100.0

    expected_subtype_set = _normalize_expected(expected_subtype)
    if not expected_subtype_set:
        raise ValueError("expected_subtype must contain at least one valid entry")

    try:
        observed_subtype = df.select(pl.col("subtype").drop_nulls().unique()).collect().get_column("subtype").to_list()
    except pl.exceptions.ColumnNotFoundError:
        observed_subtype = []
    observed_subtype_set = {v for v in observed_subtype if not (isinstance(v, int | float) and v < 0)}
    subtype_completeness = (len(expected_subtype_set & observed_subtype_set) / len(expected_subtype_set)) * 100.0
    return subtype_completeness


def role_completeness(
    df: pl.LazyFrame,
    expected_role: Sequence[betterosi.MovingObjectVehicleClassificationRole] | None,
) -> float:
    if not expected_role:
        return 100.0

    expected_role_set = _normalize_expected(expected_role)
    if not expected_role_set:
        raise ValueError("expected_role must contain at least one valid entry")

    try:
        observed_role = df.select(pl.col("role").drop_nulls().unique()).collect().get_column("role").to_list()
    except pl.exceptions.ColumnNotFoundError:
        observed_role = []
    observed_role_set = {v for v in observed_role if not (isinstance(v, int | float) and v < 0)}
    role_completeness = (len(expected_role_set & observed_role_set) / len(expected_role_set)) * 100.0
    return role_completeness


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
