"""."""

import math

import betterosi
import polars as pl

from ..metric import metric
from ...schemas import polars_schema
from .common import STATUS, PASS, FAIL, QRT

DATA_FORMAT_CONSISTENCY = "data_format_consistency"

_MAX_MOVING_OBJECT_TYPE = max(int(v) for v in betterosi.MovingObjectType)
_MAX_VEHICLE_ROLE = max(int(v) for v in betterosi.MovingObjectVehicleClassificationRole)
_MAX_VEHICLE_SUBTYPE = max(int(v) for v in betterosi.MovingObjectVehicleClassificationType)

_VALUE_CHECKS = {
    "total_nanos": pl.col("total_nanos") >= 0,
    "idx": pl.col("idx") >= 0,
    "length": pl.col("length") > 0,
    "width": pl.col("width") > 0,
    "height": pl.col("height") >= 0,
    "roll": pl.col("roll").is_between(-math.pi, math.pi),
    "pitch": pl.col("pitch").is_between(-math.pi, math.pi),
    "yaw": pl.col("yaw").is_between(-math.pi, math.pi),
    "type": pl.col("type").is_between(0, _MAX_MOVING_OBJECT_TYPE),
    "role": pl.col("role").is_between(-1, _MAX_VEHICLE_ROLE),
    "subtype": pl.col("subtype").is_between(-1, _MAX_VEHICLE_SUBTYPE),
}


@metric(computes_properties=[DATA_FORMAT_CONSISTENCY])
def data_format_consistency(df: pl.LazyFrame) -> QRT:
    schema = df.collect_schema()
    checked_columns = [column for column in polars_schema if column in schema]

    num_values = _count_non_null_values(df, checked_columns)
    num_invalid = _count_invalid_values(df, checked_columns, schema)
    value = 100.0 if num_values == 0 else 100.0 * (num_values - num_invalid) / num_values

    summary = pl.DataFrame(
        {
            DATA_FORMAT_CONSISTENCY: value,
            STATUS: [PASS if value > 95.0 else FAIL],
        }
    ).lazy()

    return df, {DATA_FORMAT_CONSISTENCY: summary}


def _count_non_null_values(df: pl.LazyFrame, columns: list[str]) -> int:
    if not columns:
        return 0

    counts = df.select([pl.col(column).count().alias(column) for column in columns]).collect().row(0)
    return sum(int(count) for count in counts)


def _count_invalid_values(df: pl.LazyFrame, columns: list[str], schema: pl.Schema) -> int:
    if not columns:
        return 0

    invalid_exprs: list[pl.Expr] = []
    for column in columns:
        if schema[column] != polars_schema[column]:
            invalid_exprs.append(pl.col(column).count().alias(column))
            continue

        value_check = _VALUE_CHECKS.get(column)
        if value_check is None:
            invalid_exprs.append(pl.lit(0).alias(column))
            continue

        invalid_exprs.append(
            pl.when(pl.col(column).is_null()).then(0).otherwise((~value_check).cast(pl.Int64)).sum().alias(column)
        )

    invalid_counts = df.select(invalid_exprs).collect().row(0)
    return sum(int(count) for count in invalid_counts)
