"""."""

import polars as pl
import pytest

from omega_prime.qualification.target_area_coverage import target_area_coverage


def test_too_few_expected_area_coords() -> None:
    df = pl.DataFrame({"x": [0.0], "y": [0.0]}).lazy()

    with pytest.raises(
        ValueError,
        match="expected_area_coords must contain at least three coordinate pairs",
    ):
        target_area_coverage(
            df,
            expected_area_coords=[(0.0, 0.0), (1.0, 1.0)],
        )


def test_invalid_expected_area_polygon() -> None:
    df = pl.DataFrame({"x": [0.0], "y": [0.0]}).lazy()

    with pytest.raises(
        ValueError,
        match="expected_area_coords does not form a valid polygon",
    ):
        target_area_coverage(
            df,
            expected_area_coords=[
                (0.0, 0.0),
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ],
        )
