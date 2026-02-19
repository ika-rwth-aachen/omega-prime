import polars as pl
import pytest

from omega_prime import Recording
from omega_prime.qualification.non_default_attr_accuracy import non_default_attributes_accuracy

from .conftest import this_assert


def test_x_is_missing(rec: Recording, capsys: pytest.CaptureFixture) -> None:
    df: pl.DataFrame = rec.df
    manipulated_df = df.drop("x")
    _df, result_dict = non_default_attributes_accuracy(manipulated_df)
    this_assert(result_dict, 65.55299539170507, False)
    stdout = capsys.readouterr().out
    assert stdout == "Column x is absent in the data frame.\n"


def test_x_and_y_are_missing(rec: Recording, capsys: pytest.CaptureFixture) -> None:
    df: pl.DataFrame = rec.df
    manipulated_df = df.drop("x", "y")
    _df, result_dict = non_default_attributes_accuracy(manipulated_df)
    this_assert(result_dict, 63.63927291346646, False)
    stdout = capsys.readouterr().out
    assert stdout == "Columns x and y are absent in the data frame.\n"


def test_x_y_and_z_are_missing(rec: Recording, capsys: pytest.CaptureFixture) -> None:
    df: pl.DataFrame = rec.df
    manipulated_df = df.drop("x", "y", "z")
    _df, result_dict = non_default_attributes_accuracy(manipulated_df)
    this_assert(result_dict, 61.50040661425861, False)
    stdout = capsys.readouterr().out
    assert stdout == "Columns x, y and z are absent in the data frame.\n"
