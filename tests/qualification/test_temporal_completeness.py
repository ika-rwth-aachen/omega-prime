"""."""

import polars as pl

from omega_prime.qualification.temporal_completeness import temporal_completeness, TEMPORAL_COMPLETENESS
from tests.qualification.conftest import qualification_assert

from .conftest import qualification_assert

def test_pass(temporal_df_pass: pl.DataFrame) -> None:
    _df, result = temporal_completeness(
        temporal_df_pass,
        expected_frequency=10.0
    )
    qualification_assert(result, TEMPORAL_COMPLETENESS, 100.0, True)


def test_fail(temporal_df_mixed: pl.DataFrame) -> None:
    _df, result = temporal_completeness(
        temporal_df_mixed,
        expected_frequency=10.0
    )
    qualification_assert(result, TEMPORAL_COMPLETENESS, 50.0, False)
