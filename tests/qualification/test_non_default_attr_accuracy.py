"""."""

from omega_prime import Recording
from omega_prime.qualification.non_default_attr_accuracy import (
    non_default_attributes_accuracy,
    NON_DEFAULT_ATTRIBUTES_ACCURACY,
)

from .conftest import qualification_assert


def test_pass(rec: Recording) -> None:
    _df, result_dict = non_default_attributes_accuracy(rec.df)
    qualification_assert(result_dict, NON_DEFAULT_ATTRIBUTES_ACCURACY, 100.0, True)
