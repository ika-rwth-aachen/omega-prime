"""."""

import polars
import pytest

from omega_prime.qualification.metric import Metric
from tests.qualification.metric.conftest import MetricFunctionSpy


def test_without_required_arg(metric: Metric) -> None:
    with pytest.raises(TypeError):
        metric.compute_lazy(polars.LazyFrame())


def test_it(spy: MetricFunctionSpy, metric: Metric) -> None:
    lf = polars.LazyFrame()
    lf_out, result_dct = metric.compute_lazy(lf, ego_id=123)
    assert id(lf_out) == id(lf)
    assert isinstance(result_dct["property_mock"], polars.LazyFrame)
    assert spy.ego_id == 123
