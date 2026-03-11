"""."""

from collections.abc import Callable

from .conftest import MetricFunctionSpy

from omega_prime.metrics.metric import metric


def test_metric_decorator(spy: MetricFunctionSpy) -> None:
    my_callable = metric(computes_properties=["property1", "property2"])
    assert isinstance(my_callable, Callable)
    obj = my_callable(spy.compute_metric)
    assert obj.computes_properties == ["property1", "property2"]
