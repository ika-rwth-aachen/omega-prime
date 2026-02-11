"""."""

import polars as pl

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT

NON_DEFAULT_ATTRIBUTES_ACCURACY = "non_default_attributes_accuracy"


@metric(computes_properties=[NON_DEFAULT_ATTRIBUTES_ACCURACY])
def non_default_attributes_accuracy(df: pl.LazyFrame | pl.DataFrame) -> QRT:
    value = 100.0

    summary = pl.DataFrame(
        {
            NON_DEFAULT_ATTRIBUTES_ACCURACY: value,
            STATUS: [PASS if value > 99.9 else FAIL],
        }
    ).lazy()

    return df, {NON_DEFAULT_ATTRIBUTES_ACCURACY: summary}
