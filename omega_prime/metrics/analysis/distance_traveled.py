import polars as pl

from ..metric import QRT, metric


@metric(computes_columns=["distance_traveled"])
def distance_traveled(df: pl.LazyFrame) -> QRT:
    """Metric that computes the column `distance_traveled`"""
    return df.with_columns(
        (pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2)
        .sqrt()
        .over("idx")
        .fill_null(0.0)
        .cum_sum()
        .alias("distance_traveled"),
    ), {}
