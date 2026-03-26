"""."""

from omega_prime.metrics.metric import Metric
from omega_prime.metrics.qualification.cli.metric_dispatch import ls_metrics, CliMetric


def test_ls_metrics() -> None:
    ls = list(CliMetric)
    assert len(ls) > 2
    metrics = ls_metrics(ls)
    for metric in metrics:
        assert isinstance(metric, Metric)
