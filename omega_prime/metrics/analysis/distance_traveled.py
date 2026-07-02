import polars as pl

from ..metric import QRT, metric


@metric(computes_columns=["distance_traveled"])
def distance_traveled(df: pl.LazyFrame) -> QRT:
    """Metric that computes the column `distance_traveled`"""
    return df.with_columns(
        (pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2)
        .sqrt()
        .fill_null(0.0)
        .cum_sum()
        .over("idx", order_by="total_nanos")
        .alias("distance_traveled"),
    ), {}
