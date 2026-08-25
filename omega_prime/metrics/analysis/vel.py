import polars as pl

from ..metric import MRT, metric


@metric(computes_columns=["vel"])
def vel(df: pl.LazyFrame) -> MRT:
    """Metric that computes the column length of the speed vecotr `vel`"""
    return df.with_columns(
        (pl.col("vel_x") ** 2 + pl.col("vel_y") ** 2).sqrt().alias("vel"),
    ), {}
