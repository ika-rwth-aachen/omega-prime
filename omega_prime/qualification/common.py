"""."""

import polars as pl

from omega_prime.schemas import polars_schema

STATUS = "status"
PASS = "pass"
FAIL = "fail"

QRT = tuple[pl.LazyFrame, dict[str, pl.LazyFrame]]


def get_num_rec(lf: pl.LazyFrame) -> int:
    return lf.select(pl.len()).collect().item(0, 0) * len(polars_schema)
