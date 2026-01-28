"""."""

import polars as pl

from ..metrics import metric


@metric(computes_properties=["duplicate_record_rate", "duplicate_record_duplicates"])
def duplicate_record_rate(
    df: pl.LazyFrame | pl.DataFrame,
    /,
    threshold: float = 1.0,
):
    group_counts = df.group_by("idx", "total_nanos").agg(pl.len().alias("record_count"))
    duplicate_detail = group_counts.filter(pl.col("record_count") > 1).with_columns(
        (pl.col("record_count") - 1).alias("duplicate_records")
    )

    total_records = df.select(pl.len().alias("total_records")).collect()[0, "total_records"]
    duplicate_records = (
        duplicate_detail.select(
            pl.col("duplicate_records").sum().fill_null(0).alias("duplicate_records")
        ).collect()[0, "duplicate_records"]
        if total_records
        else 0
    )

    duplicate_rate = (duplicate_records / total_records * 100.0) if total_records else 0.0
    status = "pass" if duplicate_rate <= threshold else "fail"

    summary = pl.DataFrame(
        {
            "total_records": [total_records],
            "duplicate_records": [duplicate_records],
            "duplicate_record_rate": [duplicate_rate],
            "threshold": [threshold],
            "status": [status],
        }
    ).lazy()

    duplicates_df = duplicate_detail.select("idx", "total_nanos", "record_count", "duplicate_records").lazy()

    return df, {
        "duplicate_record_rate": summary,
        "duplicate_record_duplicates": duplicates_df,
    }
