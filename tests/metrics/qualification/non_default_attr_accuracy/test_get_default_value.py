import polars as pl
import pytest

from omega_prime.metrics.qualification.non_default_attr_accuracy import get_default_value


def test_get_default_value():
    assert get_default_value(pl.Int64()) == 0
    assert get_default_value(pl.Float64()) == pytest.approx(0.0)
    assert get_default_value(pl.Float32()) == pytest.approx(0.0)
    assert get_default_value(pl.Binary()) == pytest.approx(b"")
    assert get_default_value(pl.String()) == pytest.approx("")

    with pytest.raises(ValueError):
        get_default_value(pl.Object())
