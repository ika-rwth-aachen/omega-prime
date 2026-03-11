import pytest

from omega_prime import Recording
from omega_prime.metrics.qualification.non_default_attr_accuracy import non_default_attributes_accuracy


def test_x_y_and_z_are_missing(rec: Recording) -> None:
    manipulated_df = rec.df.drop("x", "y", "z")
    with pytest.raises(ValueError) as err:
        non_default_attributes_accuracy(manipulated_df)
    assert err.value.args[0] == "Columns x, y and z are absent in the data frame."


def test_custom_columns_are_missing(rec: Recording) -> None:
    with pytest.raises(ValueError) as err:
        non_default_attributes_accuracy(rec.df, columns=["bogus_column", "i_wish_it_is_present"])
    ref = "Columns bogus_column and i_wish_it_is_present are absent in the data frame."
    assert err.value.args[0] == ref
