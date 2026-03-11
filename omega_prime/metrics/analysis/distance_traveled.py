@metric(computes_columns=["distance_traveled"])
def distance_traveled(df) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Metric that computes the column `distance_traveled`"""
    return df.with_columns(
        (pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2)
        .sqrt()
        .over("idx")
        .fill_null(0.0)
        .cum_sum()
        .alias("distance_traveled"),
    ), {}
