"""."""

from polars import LazyFrame

from pathlib import Path

import pytest

from omega_prime.qualification.metric import Metric, QRT


@pytest.fixture(scope="session")
def files_dir() -> Path:
    return Path(__file__).parent / "files/"


class MetricFunctionSpy:
    def __init__(self):
        self.df = LazyFrame()
        self.ego_id = -1

    def compute_metric(self, df: LazyFrame, ego_id: int) -> QRT:
        self.df = df
        self.ego_id = ego_id
        return df, {"property_mock": LazyFrame()}


@pytest.fixture
def spy() -> MetricFunctionSpy:
    return MetricFunctionSpy()


@pytest.fixture
def metric(spy: MetricFunctionSpy) -> Metric:
    return Metric(spy.compute_metric)
