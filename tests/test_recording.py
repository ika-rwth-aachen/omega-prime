import polars as pl
import pytest
from omega_prime.recording import Recording
from omega_prime.schemas import polars_schema


def test_ensure_polars_dataframe_coercion_from_dataframe():
    # Input is already a pl.DataFrame, but with Int32 and Float32 types (different from polars_schema)
    input_data = {
        "total_nanos": pl.Series([1000000000], dtype=pl.Int32),
        "idx": pl.Series([1], dtype=pl.Int32),
        "type": pl.Series([2], dtype=pl.Int32),
        "role": pl.Series([3], dtype=pl.Int32),
        "subtype": pl.Series([4], dtype=pl.Int32),
        "x": pl.Series([10.0], dtype=pl.Float32),
        "y": pl.Series([20.0], dtype=pl.Float32),
        "z": pl.Series([30.0], dtype=pl.Float32),
        "vel_x": pl.Series([1.0], dtype=pl.Float32),
        "vel_y": pl.Series([1.0], dtype=pl.Float32),
        "vel_z": pl.Series([1.0], dtype=pl.Float32),
        "acc_x": pl.Series([1.0], dtype=pl.Float32),
        "acc_y": pl.Series([1.0], dtype=pl.Float32),
        "acc_z": pl.Series([1.0], dtype=pl.Float32),
        "length": pl.Series([4.5], dtype=pl.Float32),
        "width": pl.Series([1.8], dtype=pl.Float32),
        "height": pl.Series([1.5], dtype=pl.Float32),
        "roll": pl.Series([0.0], dtype=pl.Float32),
        "pitch": pl.Series([0.0], dtype=pl.Float32),
        "yaw": pl.Series([0.0], dtype=pl.Float32),
    }

    df_in = pl.DataFrame(input_data)

    # Sanity checks on the input DataFrame types
    assert df_in.schema["total_nanos"] == pl.Int32
    assert df_in.schema["idx"] == pl.Int32
    assert df_in.schema["type"] == pl.Int32
    assert df_in.schema["role"] == pl.Int32
    assert df_in.schema["subtype"] == pl.Int32
    assert df_in.schema["x"] == pl.Float32

    # Process via the ensure method
    df_out = Recording._ensure_polars_dataframe(df_in)

    # Verify that all columns match the expected datatypes in polars_schema
    for col, expected_dtype in polars_schema.items():
        assert df_out.schema[col] == expected_dtype, f"Column {col} expected {expected_dtype}, got {df_out.schema[col]}"


def test_ensure_polars_dataframe_coercion_from_dict():
    # Input is a raw dictionary (not a pl.DataFrame) with different numeric types
    input_data = {
        "total_nanos": [1000000000],
        "idx": [1],
        "type": [2],
        "role": [3],
        "subtype": [4],
        "x": [10.0],
        "y": [20.0],
        "z": [30.0],
        "vel_x": [1.0],
        "vel_y": [1.0],
        "vel_z": [1.0],
        "acc_x": [1.0],
        "acc_y": [1.0],
        "acc_z": [1.0],
        "length": [4.5],
        "width": [1.8],
        "height": [1.5],
        "roll": [0.0],
        "pitch": [0.0],
        "yaw": [0.0],
    }

    # Process via the ensure method
    df_out = Recording._ensure_polars_dataframe(input_data)

    # Verify that the output is a Polars DataFrame and all columns have correct schema overrides/types
    assert isinstance(df_out, pl.DataFrame)
    for col, expected_dtype in polars_schema.items():
        assert df_out.schema[col] == expected_dtype, f"Column {col} expected {expected_dtype}, got {df_out.schema[col]}"
