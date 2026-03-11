@metric(computes_columns=["vel"])
def vel(df) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Metric that computes the column length of the speed vecotr `vel`"""
    return df.with_columns(
        (pl.col("vel_x") ** 2 + pl.col("vel_y") ** 2).sqrt().alias("vel"),
    ), {}
