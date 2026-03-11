from collections.abc import Callable
from dataclasses import dataclass, field
import inspect

import polars as pl

QRT = tuple[pl.LazyFrame, dict[str, pl.LazyFrame]]


@dataclass
class Metric:
    """Class to compute metrics based on polars dataframes."""

    compute_func: Callable[..., QRT]
    """The function that actually computes the metric"""
    computes_columns: list[str] = field(default_factory=list)
    """Names of the columns that will be added to the dataframe by this metrics"""
    computes_properties: list[str] = field(default_factory=list)
    """Keys of the tables added to the properties dictionary by this metric"""
    requires_columns: list[str] = field(default_factory=list)
    """Columns that must be present in the dataframe before this metric can be calculated"""
    requires_properties: list[str] = field(default_factory=list)
    """Keys of tables that must be present in the properties dictionary before this metric can be calculated."""
    computes_intermediate_columns: list[str] = field(default_factory=list)
    """Same as computes_columns by these ones will not be returned in the end and are only available to other metrics"""
    computes_intermediate_properties: list[str] = field(default_factory=list)
    """ Same as computes_properties but these ones will not be returned in the end and are only available to other metrics."""
    _parameters: list = field(init=False)
    """All parameters of metrics that need to be set on computation"""

    def get_err_msg(self, error: TypeError) -> str:
        return f"Missing parameter for Metric with compute_func {self.compute_func.__name__}: {repr(error)}"

    def compute_lazy(self, df: pl.LazyFrame, **kwargs) -> QRT:
        try:
            df, properties = self.compute_func(df, **kwargs)
            assert isinstance(df, pl.LazyFrame)
            assert all(p in properties for p in self.computes_properties + self.computes_intermediate_properties), (
                f"computes_properties {self.computes_properties} "
                f"computes_intermediate_properties {self.computes_intermediate_properties}"
            )
            return df, properties

        except TypeError as e:
            raise TypeError(self.get_err_msg(e)) from e

    def __post_init__(self):
        sig = inspect.signature(self.compute_func)
        parameters = sig.parameters
        assert "df" in parameters
        assert all(p in parameters for p in self.requires_properties)

        self._parameters = [
            v for k, v in parameters.items() if k not in ["df", "args", "kwargs"] + self.requires_properties
        ]

    def __call__(self, df: pl.DataFrame | pl.LazyFrame, **kwargs) -> QRT:
        try:
            if not isinstance(df, pl.LazyFrame):
                df = pl.LazyFrame(df)
            return self.compute_lazy(df, **kwargs)
        except TypeError as e:
            raise TypeError(self.get_err_msg(e)) from e


def metric(
    computes_columns: list[str] | None = None,
    computes_properties: list[str] | None = None,
    requires_columns: list[str] | None = None,
    requires_properties: list[str] | None = None,
    computes_intermediate_columns: list[str] | None = None,
    computes_intermediate_properties: list[str] | None = None,
):
    """Decorator to turn a function into a Metric"""

    def decorator(func):
        return Metric(
            compute_func=func,
            computes_columns=computes_columns or [],
            computes_properties=computes_properties or [],
            requires_columns=requires_columns or [],
            requires_properties=requires_properties or [],
            computes_intermediate_columns=computes_intermediate_columns or [],
            computes_intermediate_properties=computes_intermediate_properties or [],
        )

    return decorator
__all__ = ["Metric", "QRT", "metric"]
