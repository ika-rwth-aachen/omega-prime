"""."""

import polars as pl

STATUS = "status"
PASS = "pass"
FAIL = "fail"

MRT = tuple[pl.LazyFrame, dict[str, pl.LazyFrame]]


def get_num_rows(lf: pl.LazyFrame) -> int:
    return lf.select(pl.len()).collect().item(0, 0)
