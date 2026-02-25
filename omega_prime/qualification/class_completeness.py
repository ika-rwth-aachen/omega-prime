"""."""

from collections.abc import Sequence
from enum import Enum

import betterosi
import polars as pl

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT

CLASS_COMPLETENESS = "class_completeness"


@metric(computes_properties=[CLASS_COMPLETENESS])
def class_completeness(
    df: pl.LazyFrame,
    /,
    expected_classes: Sequence[betterosi.MovingObjectType],
) -> QRT:
    if not expected_classes:
        raise ValueError("expected_classes must be provided")

    expected_set = _normalize_expected(expected_classes)
    if not expected_set:
        raise ValueError("expected_classes must contain at least one valid entry")

    try:
        observed = (
            df.select(pl.col("type").drop_nulls().unique())
            .collect()
            .get_column("type")
            .to_list()
        )
    except pl.exceptions.ColumnNotFoundError:
        observed = []

    observed_set = {v for v in observed if not (isinstance(v, (int, float)) and v < 0)}
    class_completeness = (len(expected_set & observed_set) / len(expected_set)) * 100.0

    summary = pl.DataFrame(
        {
            CLASS_COMPLETENESS: class_completeness,
            STATUS: [PASS if class_completeness > 99.999999999 else FAIL],
        }
    ).lazy()

    return df, {CLASS_COMPLETENESS: summary}


def _normalize_expected(values: Sequence[betterosi.MovingObjectType]) -> set[int]:
    expected: set[int] = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, Enum):
            v = int(v)
        if isinstance(v, (int, float)) and v < 0:
            continue
        expected.add(v)
    return expected
