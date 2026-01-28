"""."""

import polars as pl

from ..metrics import metric

@metric(computes_properties=["temporal_completeness"])
def temporal_completeness(
    df,
    /,
    expected_frequency: float,
    threshold: float = 0.95,
    max_fraction_below: float = 0.05
):
    sorted_df = df.sort(["idx", "total_nanos"]).select("idx", "total_nanos")

    deltas = (
        sorted_df.with_columns((pl.col("total_nanos").diff().over("idx") / 1e9).alias("delta_t"))
        .filter(pl.col("delta_t").is_not_null())
    )

    frame_counts = df.group_by("idx").agg(pl.len().alias("sample_count"))

    delta_target = 1.0 / expected_frequency
    stats = (
        deltas.group_by("idx")
        .agg(
            pl.len().alias("observed_interval_count"),
            ((pl.col("delta_t") - delta_target) / delta_target).pow(2).mean().sqrt().alias("delta_rms"),
        )
        .with_columns(pl.lit(delta_target).alias("expected_interval"))
    )

    expected_interval_expr = pl.lit(delta_target)

    temporal_completeness = (
        frame_counts.join(stats, on="idx", how="left")
        .with_columns(
            (pl.col("sample_count") - 1).clip(lower_bound=0).alias("expected_interval_count"),
            pl.col("observed_interval_count").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("observed_interval_count") == 0)
            .then(0.0)
            .otherwise(pl.col("delta_rms"))
            .fill_null(0.0)
            .alias("delta_rms"),
            expected_interval_expr.alias("expected_interval"),
        )
        .with_columns((1 - pl.col("delta_rms")).alias("temporal_completeness"))
        .select(
            "idx",
            "sample_count",
            "expected_interval",
            "delta_rms",
            "temporal_completeness",
        )
    )

    below_expr = pl.col("temporal_completeness") < threshold
    temporal_completeness = temporal_completeness.with_columns(
        below_expr.alias("below_threshold"),
        pl.lit(threshold).alias("threshold"),
    )

    summary_frame = temporal_completeness.select(
        pl.len().alias("total_tracks"),
        pl.col("below_threshold").sum().alias("below_count"),
    ).collect()
    total_tracks = summary_frame[0, "total_tracks"]
    below_count = summary_frame[0, "below_count"]
    fraction_below = below_count / total_tracks if total_tracks else 0.0

    passes_fraction = fraction_below <= max_fraction_below
    status = "pass" if passes_fraction else "fail"

    temporal_completeness = temporal_completeness.with_columns(
        pl.lit(max_fraction_below).alias("max_fraction_below"),
        pl.lit(fraction_below).alias("fraction_below"),
        pl.lit(status).alias("status"),
    )

    return df, {"temporal_completeness": temporal_completeness}
