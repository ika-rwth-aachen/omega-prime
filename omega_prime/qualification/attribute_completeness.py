"""."""

import polars as pl
from omega_prime.schemas import polars_schema
from .common import STATUS, PASS, FAIL, QRT
from ..metrics import metric


ATTRIBUTE_COMPLETENESS = "attribute_completeness"


@metric(computes_properties=[ATTRIBUTE_COMPLETENESS])
def attribute_completeness(df: pl.LazyFrame | pl.DataFrame) -> QRT:
    schema = df.collect_schema()
    num_existing = sum(name in schema for name in polars_schema)
    attr_completeness = num_existing * 100.0 / len(polars_schema)

    summary = pl.DataFrame(
        {
            ATTRIBUTE_COMPLETENESS: attr_completeness,
            STATUS: [PASS if attr_completeness > 99.9 else FAIL],
        }
    ).lazy()

    return df, {ATTRIBUTE_COMPLETENESS: summary}
