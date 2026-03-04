"""."""

import polars as pl

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT, get_num_rows

DUPLICATE_RECORD_RATE = "duplicate_record_rate"


@metric(computes_properties=[DUPLICATE_RECORD_RATE])
def duplicate_record_rate(
    df: pl.LazyFrame,
    /,
) -> QRT:
    group_counts = df.group_by("idx", "total_nanos").agg(pl.len().alias("record_count"))

    total_records = get_num_rows(df)
    duplicate_records_per_group = (
        pl.when(pl.col("record_count") > 1)
        .then(pl.col("record_count") - 1)
        .otherwise(0)
        .alias("duplicate_records_per_group")
    )
    duplicate_records = group_counts.select(
        duplicate_records_per_group.sum().fill_null(0).alias("duplicate_records")
    ).collect()[0, "duplicate_records"]

    duplicate_rate = (duplicate_records / total_records * 100.0) if total_records else 0.0
    status = PASS if duplicate_rate <= 1.0 else FAIL

    summary = pl.DataFrame(
        {
            DUPLICATE_RECORD_RATE: duplicate_rate,
            STATUS: [status],
        }
    ).lazy()

    return df, {DUPLICATE_RECORD_RATE: summary}
