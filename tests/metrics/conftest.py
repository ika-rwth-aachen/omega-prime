"""."""

from pathlib import Path

import pytest
from polars import LazyFrame

import omega_prime

from omega_prime.metrics.metric import Metric, QRT


@pytest.fixture(scope="session")
def files_dir() -> Path:
    return Path(__file__).parent.parent / "files/"


@pytest.fixture(scope="session")
def rec(files_dir: Path):
    return omega_prime.Recording.from_file(str(files_dir / "pedestrian.osi"))


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
