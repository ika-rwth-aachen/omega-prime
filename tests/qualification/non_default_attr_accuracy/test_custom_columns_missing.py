import pytest

from omega_prime import Recording
from omega_prime.qualification.non_default_attr_accuracy import non_default_attributes_accuracy


def test_x_is_missing(rec: Recording) -> None:
    manipulated_df1 = rec.df.drop("x")
    with pytest.raises(ValueError) as err:
        non_default_attributes_accuracy(manipulated_df1)
    assert err.value.args[0] == "Column x is absent in the data frame."


def test_x_and_y_are_missing(rec: Recording) -> None:
    manipulated_df = rec.df.drop("x", "y")
    with pytest.raises(ValueError) as err:
        non_default_attributes_accuracy(manipulated_df)
    assert err.value.args[0] == "Columns x and y are absent in the data frame."


def test_x_y_and_z_are_missing(rec: Recording) -> None:
    manipulated_df = rec.df.drop("x", "y", "z")
    with pytest.raises(ValueError) as err:
        non_default_attributes_accuracy(manipulated_df)
    assert err.value.args[0] == "Columns x, y and z are absent in the data frame."
