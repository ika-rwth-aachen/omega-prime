"""."""

from omega_prime import Recording
from omega_prime.qualification.non_default_attr_accuracy import non_default_attributes_accuracy

from .conftest import this_assert


def test_fail_with_default_columns(rec: Recording) -> None:
    """Without any keyword arguments, the column names are taken from the polars_schema."""
    _df, result_dict = non_default_attributes_accuracy(rec.df)
    this_assert(result_dict, 67.27534562211981, False)
