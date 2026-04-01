"""."""

import typer.models

from omega_prime.metrics.metric import Metric
from omega_prime.metrics.metric_manager import MetricManager
from ..attribute_completeness import attribute_completeness
from ..class_completeness import class_completeness
from ..data_format_consistency import data_format_consistency
from ..duplicate_record_rate import duplicate_record_rate
from ..non_default_attr_accuracy import non_default_attributes_accuracy
from ..object_type_coverage import object_type_coverage
from ..record_completeness import record_completeness
from ..target_area_coverage import target_area_coverage
from ..temporal_completeness import temporal_completeness
from ..temporal_coverage import temporal_coverage

from enum import Enum


class CliMetric(Enum):
    """These are the metrics exposed to the user via CLI."""

    RECORD_COMPLETENESS = "record-completeness"
    ATTRIBUTE_COMPLETENESS = "attribute-completeness"
    CLASS_COMPLETENESS = "class-completeness"
    DATA_FORMAT_CONSISTENCY = "data-format-consistency"
    DUPLICATE_RECORD_RATE = "duplicate-record-rate"
    TEMPORAL_COMPLETENESS = "temporal-completeness"
    TEMPORAL_COVERAGE = "temporal-coverage"
    OBJECT_TYPE_COVERAGE = "object-type-coverage"
    NON_DEFAULT_ATTRIBUTES_ACCURACY = "non-default-attributes-accuracy"
    TARGET_AREA_COVERAGE = "target-area-coverage"

    @classmethod
    def get_default(cls) -> tuple["CliMetric", "CliMetric"]:
        return cls.CLASS_COMPLETENESS, cls.OBJECT_TYPE_COVERAGE

    @staticmethod
    def get_option() -> typer.models.OptionInfo:
        return typer.Option("--metric", "-m", help="Select one or more metrics.")


MAP_METRIC_NAME = {
    CliMetric.RECORD_COMPLETENESS.value: record_completeness,
    CliMetric.ATTRIBUTE_COMPLETENESS.value: attribute_completeness,
    CliMetric.CLASS_COMPLETENESS.value: class_completeness,
    CliMetric.DATA_FORMAT_CONSISTENCY.value: data_format_consistency,
    CliMetric.DUPLICATE_RECORD_RATE.value: duplicate_record_rate,
    CliMetric.TEMPORAL_COMPLETENESS.value: temporal_completeness,
    CliMetric.TEMPORAL_COVERAGE.value: temporal_coverage,
    CliMetric.OBJECT_TYPE_COVERAGE.value: object_type_coverage,
    CliMetric.NON_DEFAULT_ATTRIBUTES_ACCURACY.value: non_default_attributes_accuracy,
    CliMetric.TARGET_AREA_COVERAGE.value: target_area_coverage,
}


def ls_metrics(cli_metrics: list[CliMetric]) -> list[Metric]:
    return [MAP_METRIC_NAME[cm.value] for cm in cli_metrics]


def get_metric_manager(cli_metrics: list[CliMetric]) -> MetricManager:
    metrics = ls_metrics(cli_metrics)
    return MetricManager(metrics=metrics)
