"""."""

from omega_prime.metrics.qualification.cli.metric_dispatch import get_metric_manager, CliMetric
from omega_prime.metrics.metric_manager import MetricManager


def test_get_metric_manager() -> None:
    mm = get_metric_manager([CliMetric.CLASS_COMPLETENESS, CliMetric.OBJECT_TYPE_COVERAGE])
    assert isinstance(mm, MetricManager)
    assert len(mm.metrics) == 2
