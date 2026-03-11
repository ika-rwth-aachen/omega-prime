"""."""

from omega_prime.qualification.metric import Metric

from .conftest import MetricFunctionSpy


def test_init(spy: MetricFunctionSpy, metric: Metric) -> None:
    assert metric.compute_func == spy.compute_metric
    assert len(metric._parameters) == 1
    assert metric._parameters[0].name == "ego_id"
