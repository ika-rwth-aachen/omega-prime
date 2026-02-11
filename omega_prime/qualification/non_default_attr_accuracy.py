"""."""

from collections.abc import Sequence

import polars as pl

from ..metrics import metric
from ..schemas import polars_schema
from .common import STATUS, PASS, FAIL, QRT, get_num_rec

NON_DEFAULT_ATTRIBUTES_ACCURACY = "non_default_attributes_accuracy"


def get_column_names(t: type[int] | type[float]) -> list[str]:
    """List of column names of a given type.

    Args:
        t: should be `int` or `float`.

    Returns:
        list of column names selected from the schema.
    """
    pl_type = pl.Int64 if t is int else pl.Float64
    return [k for k, v in polars_schema.items() if v == pl_type]


@metric(computes_properties=[NON_DEFAULT_ATTRIBUTES_ACCURACY])
def non_default_attributes_accuracy(df: pl.LazyFrame | pl.DataFrame, /, columns: Sequence[str] = tuple()) -> QRT:
    """Non default valued attributes accuracy.

        Measures how often attributes contain non-default values, reflecting data richness.
        Flags overused defaults (e.g., unknown).
        Formula is (number of non-default value) * 100 / (total number of values)
        Threshold is greater or equal than 95%.

    Args:
        df: omega-prime data frame (LazyFrame).
        columns: columns to query. If not given, the schema keys are used.

    Returns:
        The tuple of original data and qualification summary (dictionary).
    """
    cc = columns if columns else polars_schema.keys()
    s_cc = df.select(cc)
    select = s_cc.select(pl.sum_horizontal(*[pl.col(c) == 0 for c in cc]).sum())
    num_def = int(select.collect().item(0, 0))

    num_rec = get_num_rec(df)
    value = 100.0 * (num_rec - num_def) / num_rec

    summary = pl.DataFrame(
        {
            NON_DEFAULT_ATTRIBUTES_ACCURACY: value,
            STATUS: [PASS if value > 95.0 else FAIL],
        }
    ).lazy()

    return df, {NON_DEFAULT_ATTRIBUTES_ACCURACY: summary}
