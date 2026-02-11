"""."""

import polars as pl

STATUS = "status"
PASS = "pass"
FAIL = "fail"

QRT = tuple[pl.LazyFrame, dict[str, pl.LazyFrame]]
