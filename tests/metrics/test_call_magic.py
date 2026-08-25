"""."""

import polars
import pytest

from .conftest import MetricFunctionSpy

from omega_prime.metrics.metric import Metric


def test_it(spy: MetricFunctionSpy, metric: Metric) -> None:
    df_out, result_dct = metric(polars.DataFrame(), ego_id=123)
    assert isinstance(df_out, polars.LazyFrame)
    assert isinstance(result_dct["property_mock"], polars.LazyFrame)
    assert spy.ego_id == 123


def test_without_required_parameter(metric: Metric) -> None:
    with pytest.raises(TypeError):
        metric(polars.DataFrame())
