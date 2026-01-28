"""."""

from collections.abc import Sequence
from enum import Enum
from typing import Any

import betterosi
import polars as pl

from omega_prime import Recording

from ..metrics import metric


@metric(computes_properties=["class_completeness"])
def class_completeness(
    df,
    /,
    expected_classes: Sequence[Any] | None = None,
    expected_subtypes: Sequence[Any] | None = None,
    expected_roles: Sequence[Any] | None = None,
    expected_tl_colors: Sequence[Any] | None = None,
    expected_tl_icons: Sequence[Any] | None = None,
    expected_tl_modes: Sequence[Any] | None = None,
    threshold: float = 95.0,
    allow_missing: int = 0,
    traffic_light_states: dict | None = None,
    recording: Recording | None = None,
):
    if traffic_light_states is None and recording is not None:
        traffic_light_states = getattr(recording, "traffic_light_states", None)

    type_names = (
        _prepare_distinct_names(_collect_unique(df, "type"), betterosi.MovingObjectType)
        if expected_classes
        else []
    )
    subtype_names = (
        _prepare_distinct_names(
            _collect_unique(df, "subtype"),
            betterosi.MovingObjectVehicleClassificationType,
        )
        if expected_subtypes
        else []
    )
    role_names = (
        _prepare_distinct_names(
            _collect_unique(df, "role"),
            betterosi.MovingObjectVehicleClassificationRole,
        )
        if expected_roles
        else []
    )

    color_names: list[str] = []
    icon_names: list[str] = []
    mode_names: list[str] = []

    def build_summary(
        check: str,
        field_name: str,
        observed_names: set[str],
        expected_values: Sequence[Any] | None,
        enum_cls: type[Enum],
    ) -> dict[str, Any]:
        expected_set = _normalize_expected(expected_values, enum_cls)
        if expected_values is None or len(expected_values) == 0:
            return {
                "check": check,
                "field": field_name,
                "expected_count": None,
                "present_count": len(observed_names),
                "coverage": None,
                "threshold": None,
                "allow_missing": None,
                "missing_classes": None,
                "unexpected_classes": None,
                "observed_classes": sorted(observed_names),
                "passes_threshold": None,
                "passes_with_allowance": None,
                "status": "not_configured",
            }
        if not expected_set:
            return {
                "check": check,
                "field": field_name,
                "expected_count": 0,
                "present_count": len(observed_names),
                "coverage": None,
                "threshold": threshold,
                "allow_missing": allow_missing,
                "missing_classes": [],
                "unexpected_classes": sorted(observed_names),
                "observed_classes": sorted(observed_names),
                "passes_threshold": None,
                "passes_with_allowance": None,
                "status": "not_configured",
            }

        coverage = len(observed_names & expected_set) * 100.0 / len(expected_set)
        missing_classes = sorted(expected_set - observed_names)
        unexpected_classes = sorted(observed_names - expected_set)
        passes_threshold = coverage >= threshold
        passes_with_allowance = len(missing_classes) <= allow_missing
        status = "pass" if (passes_threshold or passes_with_allowance) else "fail"
        return {
            "check": check,
            "field": field_name,
            "expected_count": len(expected_set),
            "present_count": len(observed_names),
            "coverage": coverage,
            "threshold": threshold,
            "allow_missing": allow_missing,
            "missing_classes": missing_classes,
            "unexpected_classes": unexpected_classes,
            "observed_classes": sorted(observed_names),
            "passes_threshold": passes_threshold,
            "passes_with_allowance": passes_with_allowance,
            "status": status,
        }

    summary_rows: list[dict[str, Any]] = []

    if expected_classes:
        summary_rows.append(
            build_summary(
                check="type_coverage",
                field_name="moving_object.type",
                observed_names=set(type_names),
                expected_values=expected_classes,
                enum_cls=betterosi.MovingObjectType,
            )
        )

    if expected_subtypes:
        summary_rows.append(
            build_summary(
                check="subtype_coverage",
                field_name="moving_object.vehicle_classification.type",
                observed_names=set(subtype_names),
                expected_values=expected_subtypes,
                enum_cls=betterosi.MovingObjectVehicleClassificationType,
            )
        )

    if expected_roles:
        summary_rows.append(
            build_summary(
                check="role_coverage",
                field_name="moving_object.vehicle_classification.role",
                observed_names=set(role_names),
                expected_values=expected_roles,
                enum_cls=betterosi.MovingObjectVehicleClassificationRole,
            )
        )

    include_tl_checks = any(seq for seq in (expected_tl_colors, expected_tl_icons, expected_tl_modes))
    if include_tl_checks:
        tl_states = traffic_light_states or {}
        color_names = _collect_traffic_light_classes(tl_states, "color", betterosi.TrafficLightClassificationColor)
        icon_names = _collect_traffic_light_classes(tl_states, "icon", betterosi.TrafficLightClassificationIcon)
        mode_names = _collect_traffic_light_classes(tl_states, "mode", betterosi.TrafficLightClassificationMode)
        if expected_tl_colors:
            summary_rows.append(
                build_summary(
                    check="traffic_light_color_coverage",
                    field_name="traffic_light.classification.color",
                    observed_names=set(color_names),
                    expected_values=expected_tl_colors,
                    enum_cls=betterosi.TrafficLightClassificationColor,
                )
            )

        if expected_tl_icons:
            summary_rows.append(
                build_summary(
                    check="traffic_light_icon_coverage",
                    field_name="traffic_light.classification.icon",
                    observed_names=set(icon_names),
                    expected_values=expected_tl_icons,
                    enum_cls=betterosi.TrafficLightClassificationIcon,
                )
            )

        if expected_tl_modes:
            summary_rows.append(
                build_summary(
                    check="traffic_light_mode_coverage",
                    field_name="traffic_light.classification.mode",
                    observed_names=set(mode_names),
                    expected_values=expected_tl_modes,
                    enum_cls=betterosi.TrafficLightClassificationMode,
                )
            )

    if not summary_rows:
        raise ValueError("class_completeness requires at least one expected_* list with entries")

    summary = pl.DataFrame(summary_rows).lazy()

    return df, {"class_completeness": summary}


def _collect_traffic_light_classes(
    traffic_light_states: dict | None,
    attr: str,
    enum_cls: type[Enum],
) -> list[str]:
    distinct: set[str] = set()
    if not traffic_light_states:
        return []
    for tl_list in traffic_light_states.values():
        for tl in tl_list:
            classification = getattr(tl, "classification", None)
            if classification is None:
                continue
            value = getattr(classification, attr, None)
            if value is None:
                continue
            name = _enum_name(enum_cls, value)
            if name is None:
                continue
            distinct.add(name)
    return sorted(distinct)

def _collect_unique(df: pl.LazyFrame, column: str) -> list[Any]:
    try:
        return df.select(pl.col(column).drop_nulls().unique()).collect()[column].to_list()
    except pl.exceptions.ColumnNotFoundError:
        return []

def _enum_name(enum_cls: type[Enum], value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        enum_value = value
    else:
        try:
            enum_value = enum_cls(value)
        except Exception:
            return str(value)

    if "UNKNOWN" in enum_value.name.upper():
        return None
    return enum_value.name

def _normalize_expected(values: Sequence[Any] | None, enum_cls: type[Enum]) -> set[str]:
    if not values:
        return set()
    expected: set[str] = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            expected.add(v)
        elif isinstance(v, Enum):
            name = _enum_name(type(v), v)
            if name is not None:
                expected.add(name)
        else:
            name = _enum_name(enum_cls, v)
            if name is not None:
                expected.add(name)
    return expected

def _prepare_distinct_names(values: Sequence[Any], enum_cls: type[Enum] | None = None) -> list[str]:
    distinct: set[str] = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, (int, float)) and v < 0:
            continue
        if enum_cls is not None:
            name = _enum_name(enum_cls, v)
            if name is None:
                continue
            distinct.add(name)
        elif isinstance(v, Enum):
            name = _enum_name(type(v), v)
            if name is None:
                continue
            distinct.add(name)
        else:
            distinct.add(str(v))
    return sorted(distinct)
