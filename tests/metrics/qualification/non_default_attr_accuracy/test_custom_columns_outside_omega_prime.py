import pytest

from omega_prime import Recording
from omega_prime.metrics.qualification.non_default_attr_accuracy import non_default_attributes_accuracy
from .conftest import this_assert


def test_known_default(rec: Recording) -> None:
    _df, result_dict = non_default_attributes_accuracy(rec.df, columns=("geometry",))
    this_assert(result_dict, 100.0, True)


def test_unknown_default(rec: Recording) -> None:
    with pytest.raises(ValueError):
        non_default_attributes_accuracy(rec.df, columns=("polygon",))
