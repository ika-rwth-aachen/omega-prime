"""."""

import pytest
import polars as pl

from omega_prime.qualification.temporal_completeness import temporal_completeness

def test_pass(temporal_df_pass: pl.DataFrame) -> None:
    _df, result_dict = temporal_completeness(
        temporal_df_pass,
        expected_frequency=10.0,
        threshold=0.8,
        max_fraction_below=0.2,
    )
    tc_df = result_dict["temporal_completeness"].collect()
    assert tc_df["fraction_below"][0] == pytest.approx(0.0)
    assert tc_df["status"][0] == "pass"

def test_fail(temporal_df_mixed: pl.DataFrame) -> None:
    _df, result_dict = temporal_completeness(
        temporal_df_mixed,
        expected_frequency=10.0,
        threshold=0.95,
        max_fraction_below=0.0,
    )
    tc_df = result_dict["temporal_completeness"].collect()
    assert tc_df["fraction_below"][0] == pytest.approx(0.5)
    assert tc_df["status"][0] == "fail"
