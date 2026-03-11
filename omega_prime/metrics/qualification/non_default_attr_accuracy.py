"""."""

from collections.abc import Sequence

import polars as pl

from .english_syntax import get_is_ending, format_items
from ..metric import metric
from ..schemas import polars_schema
from .common import STATUS, PASS, FAIL, QRT, get_num_rows

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


POLARS_T = pl.Int64 | pl.Float64 | pl.Float32 | pl.Binary | pl.Object | pl.String


def get_default_value(polars_type: POLARS_T) -> float | int | str | bytes:
    x = 0  # silencing the code checker in PyCharm.
    # fmt: off
    match polars_type:
        case pl.Int64: x = 0
        case pl.Float64 | pl.Float32: x = 0.0
        case pl.Binary: x = b''
        case pl.String: x = ''
        case _:
            raise ValueError(f'Default value for type {polars_type} is undefined')
    # fmt: on
    return x


@metric(computes_properties=[NON_DEFAULT_ATTRIBUTES_ACCURACY])
def non_default_attributes_accuracy(df: pl.LazyFrame, /, columns: Sequence[str] = tuple()) -> QRT:
    """Non default valued attributes accuracy.

        Measures how often attributes contain non-default values, reflecting data richness.
        Flags overused defaults (e.g., unknown).
        Formula is (number of non-default value) * 100 / (total number of values)
        Threshold is greater or equal than 95%.

        Design choices: the keyword argument `columns` allows to analyze a custom set of columns,
        even those outside the OmegaPrime data schema `polars_schema`.
        The default values for the columns outside the OmegaPrime data schema are derived
        from the schema of the data frame.

    Args:
        df: omega-prime data frame (LazyFrame).
        columns: columns to query. If not given, the schema keys are used.

    Returns:
        The tuple of original data and qualification summary (dictionary).

    Raises:
        ValueError: if some columns could not be found.

    """
    schema = df.collect_schema()
    cc = set(columns) if columns else polars_schema.keys()
    absent_cols = set(c for c in cc if c not in schema)
    if absent_cols:
        is_verb, ending = get_is_ending(len(absent_cols))
        items = format_items(sorted(absent_cols))
        raise ValueError(f"Column{ending} {items} {is_verb} absent in the data frame.")

    s_cc = df.select(cc)
    gen_expr = (pl.col(c) == get_default_value(schema[c]) for c in cc)
    select = s_cc.select(pl.sum_horizontal(*gen_expr).sum())
    num_def = int(select.collect().item(0, 0))

    num_rec = get_num_rows(df) * len(cc)
    value = 100.0 * (num_rec - num_def) / num_rec

    summary = pl.DataFrame(
        {
            NON_DEFAULT_ATTRIBUTES_ACCURACY: value,
            STATUS: [PASS if value > 95.0 else FAIL],
        }
    ).lazy()

    return df, {NON_DEFAULT_ATTRIBUTES_ACCURACY: summary}
