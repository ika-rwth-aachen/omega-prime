"""."""

from omega_prime.metrics.qualification.cli.metric_dispatch import CliMetric


def test_get_default() -> None:
    assert CliMetric.get_default() == (CliMetric.CLASS_COMPLETENESS, CliMetric.OBJECT_TYPE_COVERAGE)
