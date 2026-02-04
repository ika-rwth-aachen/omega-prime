"""."""

import polars as pl
from omega_prime.schemas import polars_schema
from .common import STATUS, PASS, FAIL, QRT
from ..metrics import metric

RECORD_COMPLETENESS = "record_completeness"


@metric(computes_properties=[RECORD_COMPLETENESS])
def record_completeness(df: pl.LazyFrame) -> QRT:
    num_rec = df.select(pl.len()).collect().item(0, 0) * len(polars_schema)
    num_nil = df.select_seq(polars_schema.keys()).count().collect().sum_horizontal().item(0)
    rec_completeness = num_nil * 100.0 / num_rec

    summary = pl.DataFrame(
        {
            RECORD_COMPLETENESS: rec_completeness,
            STATUS: [PASS if rec_completeness > 95.0 else FAIL],
        }
    ).lazy()

    return df, {RECORD_COMPLETENESS: summary}
