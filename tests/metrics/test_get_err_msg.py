"""."""

from omega_prime.qualification.metric import Metric


def test_it(metric: Metric) -> None:
    ref = "Missing parameter for Metric with compute_func compute_metric: TypeError()"
    assert metric.get_err_msg(TypeError()) == ref
