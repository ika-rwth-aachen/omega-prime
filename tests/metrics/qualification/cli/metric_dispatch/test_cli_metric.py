"""."""

from omega_prime.metrics.qualification.cli.metric_dispatch import CliMetric


def test_get_default() -> None:
    assert CliMetric.get_default() == (CliMetric.CLASS_COMPLETENESS, CliMetric.OBJECT_TYPE_COVERAGE)


def test_registered_metrics() -> None:
    assert {metric.value for metric in CliMetric} == {
        "record-completeness",
        "attribute-completeness",
        "class-completeness",
        "data-format-consistency",
        "duplicate-record-rate",
        "temporal-completeness",
        "temporal-coverage",
        "object-type-coverage",
        "non-default-attributes-accuracy",
        "target-area-coverage",
    }
