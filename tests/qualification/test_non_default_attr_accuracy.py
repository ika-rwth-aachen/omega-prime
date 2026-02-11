"""."""

from omega_prime import Recording
from omega_prime.qualification.non_default_attr_accuracy import (
    non_default_attributes_accuracy,
    NON_DEFAULT_ATTRIBUTES_ACCURACY,
    get_column_names,
)

from .conftest import qualification_assert


def test_get_column_names() -> None:
    assert get_column_names(int) == ["total_nanos", "idx", "type", "role", "subtype"]
    # fmt: off
    ref = [
        'x', 'y', 'z',
        'vel_x', 'vel_y', 'vel_z',
        'acc_x', 'acc_y', 'acc_z',
        'length', 'width', 'height',
        'roll', 'pitch', 'yaw'
    ]
    # fmt: on
    assert get_column_names(float) == ref


def test_fail_with_default_columns(rec: Recording) -> None:
    """Without any keyword arguments, the column names are taken from the polars_schema."""
    _df, result_dict = non_default_attributes_accuracy(rec.df)
    qualification_assert(result_dict, NON_DEFAULT_ATTRIBUTES_ACCURACY, 67.27534562211981, False)


def test_pass_for_xyz(rec: Recording) -> None:
    _df, result_dict = non_default_attributes_accuracy(rec.df, columns=("x", "y", "z"))
    qualification_assert(result_dict, NON_DEFAULT_ATTRIBUTES_ACCURACY, 100.0, True)


def test_pass_for_all_float(rec: Recording) -> None:
    _df, result_dict = non_default_attributes_accuracy(rec.df, columns=get_column_names(float))
    qualification_assert(result_dict, NON_DEFAULT_ATTRIBUTES_ACCURACY, 72.286866359447, False)


def test_pass_for_all_int(rec: Recording) -> None:
    _df, result_dict = non_default_attributes_accuracy(rec.df, columns=get_column_names(int))
    qualification_assert(result_dict, NON_DEFAULT_ATTRIBUTES_ACCURACY, 94.98847926267281, False)
