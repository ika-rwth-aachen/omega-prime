"""."""

import polars as pl

from ..metrics import metric
from .common import STATUS, PASS, FAIL, QRT, get_num_rows

DUPLICATE_RECORD_RATE = "duplicate_record_rate"
DUPLICATE_RECORD_DUPLICATES = "duplicate_record_duplicates"


@metric(computes_properties=[DUPLICATE_RECORD_RATE, DUPLICATE_RECORD_DUPLICATES])
def duplicate_record_rate(
    df: pl.LazyFrame,
    /,
    threshold: float = 1.0,
) -> QRT:
    group_counts = df.group_by("idx", "total_nanos").agg(pl.len().alias("record_count"))
    duplicate_detail = group_counts.filter(pl.col("record_count") > 1).with_columns(
        (pl.col("record_count") - 1).alias("duplicate_records")
    )

    total_records = get_num_rows(df)
    duplicate_records = (
        duplicate_detail.select(pl.col("duplicate_records").sum().fill_null(0).alias("duplicate_records")).collect()[
            0, "duplicate_records"
        ]
        if total_records
        else 0
    )

    duplicate_rate = (duplicate_records / total_records * 100.0) if total_records else 0.0
    status = PASS if duplicate_rate <= threshold else FAIL

    summary = pl.DataFrame(
        {
            DUPLICATE_RECORD_RATE: duplicate_rate,
            "total_records": [total_records],
            "duplicate_records": [duplicate_records],
            "threshold": [threshold],
            STATUS: [status],
        }
    ).lazy()

    duplicates_df = duplicate_detail.select("idx", "total_nanos", "record_count", "duplicate_records").lazy()

    return df, {
        DUPLICATE_RECORD_RATE: summary,
        DUPLICATE_RECORD_DUPLICATES: duplicates_df,
    }
