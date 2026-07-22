from omega_prime import Recording
from omega_prime.metrics.qualification.non_default_attr_accuracy import (
    get_column_names,
    non_default_attributes_accuracy,
)
from .conftest import this_assert


def test_pass_for_xyz(rec: Recording) -> None:
    _df, result_dict = non_default_attributes_accuracy(rec.df, columns=("x", "y", "z"))
    this_assert(result_dict, 100.0, True)


def test_fail_for_all_float(rec: Recording) -> None:
    _df, result_dict = non_default_attributes_accuracy(rec.df, columns=get_column_names(float))
    this_assert(result_dict, 63.04915514592934, False)


def test_fail_for_all_int(rec: Recording) -> None:
    _df, result_dict = non_default_attributes_accuracy(rec.df, columns=get_column_names(int))
    this_assert(result_dict, 79.95391705069125, False)
