import polars as pl
from dataclasses import dataclass, field
from collections.abc import Callable
import polars_st as st
from .recording import Recording
import graphlib
import inspect


@dataclass
class Metric:
    compute_func: Callable[[pl.LazyFrame, ...], tuple[pl.LazyFrame, dict[str, pl.LazyFrame]]]
    computes_columns: list[str] = field(default_factory=list)
    computes_properties: list[str] = field(default_factory=list)
    requires_columns: list[str] = field(default_factory=list)
    requires_properties: list[str] = field(default_factory=list)
    computes_intermediate_columns: list[str] = field(default_factory=list)
    computes_intermediate_properties: list[str] = field(default_factory=list)
    _parameters: list = field(init=False)

    def compute_lazy(self, df: pl.LazyFrame, **kwargs) -> tuple[pl.LazyFrame, dict[str, pl.LazyFrame]]:
        try:
            df, properties = self.compute_func(df, **kwargs)
            assert isinstance(df, pl.LazyFrame)
            assert all(p in properties for p in self.computes_properties+self.computes_intermediate_properties)
            return df, properties

        except TypeError as e:
            raise TypeError(
                f"Missing paramter for Metric with compute_func {self.compute_func.__name__}: {repr(e)}"
            ) from e

    def __post_init__(self):
        sig = inspect.signature(self.compute_func)
        parameters = sig.parameters
        assert "df" in parameters
        assert all(p in parameters for p in self.requires_properties)

        self._parameters = [
            v for k, v in parameters.items() if k not in ["df", "args", "kwargs"] + self.requires_properties
        ]

    def __call__(self, df: pl.DataFrame, **kwargs):
        try:
            if not isinstance(df, pl.LazyFrame):
                df = pl.LazyFrame(df)
            return self.compute_lazy(df, **kwargs)
        except TypeError as e:
            raise TypeError(
                f"Missing paramter for Metric with compute_func {self.compute_func.__name__}: {repr(e)}"
            ) from e


def metric(
    computes_columns: list[str] | None = None,
    computes_properties: list[str] | None = None,
    requires_columns: list[str] | None = None,
    requires_properties: list[str] | None = None,
    computes_intermediate_columns: list[str] | None = None,
    computes_intermediate_properties: list[str] | None = None
):
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


@metric(computes_columns=["distance_traveled"])
def distance_traveled(df) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    return df.with_columns(
        (pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2)
        .sqrt()
        .over("idx")
        .fill_null(0.0)
        .cum_sum()
        .alias("distance_traveled"),
        
    ), {}

@metric(computes_columns=['vel'])
def vel(df) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    return df.with_columns(
        (pl.col("vel_x") ** 2 + pl.col("vel_y") ** 2).sqrt().alias("vel"),
    ), {}


@metric(
    requires_columns=["distance_traveled", "vel"],
    computes_properties=["timegaps", "min_timegaps"],
    computes_intermediate_properties=['crossed']
)
def timegaps_and_min_timgaps(df, /, ego_id, time_buffer=2e9):
    ego_df = df.filter(idx=ego_id)

    crossed = df.join(ego_df, how="cross", suffix="_ego")

    crossed = crossed.filter(
        (pl.col("total_nanos_ego") - time_buffer) <= pl.col("total_nanos"),
        (pl.col("total_nanos_ego") + time_buffer) >= pl.col("total_nanos"),
        pl.col("idx_ego") != pl.col("idx"),
    )

    all_timegaps = (
        crossed.filter(pl.col("geometry").st.intersects(pl.col("geometry_ego")))
        .with_columns(timegap=(pl.col("total_nanos") - pl.col("total_nanos_ego")) / 1e9)
        .select(
            "idx_ego", "idx", "total_nanos_ego", "total_nanos", "timegap", "distance_traveled", "distance_traveled_ego"
        )
    )

    timegaps = (
        all_timegaps.group_by("idx", "idx_ego", "total_nanos_ego")
        .agg(
            pl.col("timegap", "total_nanos", "distance_traveled", "distance_traveled_ego").get(
                pl.col("timegap").abs().arg_min()
            ),
        )
        .sort("idx_ego", "idx", "total_nanos_ego")
        .select(
            "idx_ego", "idx", "total_nanos_ego", "timegap", "total_nanos", "distance_traveled", "distance_traveled_ego"
        )
    )
    min_timegaps = timegaps.group_by("idx_ego", "idx").agg(
        pl.col("timegap").get(pl.col("timegap").abs().arg_min()).alias("min_timegap")
    )

    return df, {
        "timegaps": timegaps,
        "min_timegaps": min_timegaps,
        "crossed": crossed
    }
    

@metric(
    requires_columns=["distance_traveled", "vel"],
    requires_properties=['crossed','timegaps'],
    computes_properties=["p_timegaps", "min_p_timegaps"],
)
def p_timegaps_and_min_p_timgaps(df, /, ego_id, crossed, timegaps,time_buffer=2e9):
    p_timegaps = (
        crossed.join(timegaps, how="right", suffix="_overlap", on=["idx", "idx_ego"])
        .with_columns(
            pl.when(pl.col("total_nanos") >= pl.col("total_nanos_overlap"))
            .then((pl.col("total_nanos_overlap") - pl.col("total_nanos")) / 1e9)
            .otherwise((pl.col("distance_traveled_overlap") - pl.col("distance_traveled")) / pl.col("vel"))
            .alias("time_to_overlap"),
            pl.when(pl.col("total_nanos_ego") >= pl.col("total_nanos_ego_overlap"))
            .then((pl.col("total_nanos_ego_overlap") - pl.col("total_nanos_ego")) / 1e9)
            .otherwise((pl.col("distance_traveled_ego_overlap") - pl.col("distance_traveled_ego")) / pl.col("vel_ego"))
            .alias("time_to_overlap_ego"),
        )
        .with_columns(
            -(
                pl.col("time_to_overlap_ego")
                - pl.col("time_to_overlap")
                + (pl.col("total_nanos_ego") - pl.col("total_nanos")) / 1e9
            ).alias("p_timegap")
        )
        .group_by("idx_ego", "idx", "total_nanos_ego")
        .agg(
            pl.col("p_timegap", "total_nanos")
            .sort_by(pl.col("p_timegap").abs(), descending=False, nulls_last=True)
            .first()
        )
        .sort("idx_ego", "idx", "total_nanos_ego")
    )

    min_p_timegaps = p_timegaps.group_by("idx_ego", "idx").agg(
        pl.col("p_timegap").sort_by(pl.col("p_timegap").abs(), descending=False).first()
    )

    return df, {
        "p_timegaps": p_timegaps,
        "min_p_timegaps": min_p_timegaps,
    }


metrics = [
    vel,
    distance_traveled,
    timegaps_and_min_timgaps, 
    p_timegaps_and_min_p_timgaps
]


@dataclass
class MetricManager:
    metrics: list[Metric] = field(default_factory=lambda: metrics)
    exclude_columns: list[str] = field(default_factory=list)
    exclude_properties: list[str] = field(default_factory=list)
    _dependencies: dict[int | str, list[int | str]] = field(init=False)
    _ordered_metrics: list[Metric] = field(init=False)
    _parameters: list = field(init=False)

    def __post_init__(self):
        self._dependencies = {
            val: [i]
            for i, m in enumerate(self.metrics)
            for val in [f"column_{n}" for n in m.computes_columns+m.computes_intermediate_columns] + [f"property_{n}" for n in m.computes_properties+m.computes_intermediate_properties]
        } | {
            i: [f"column_{n}" for n in m.requires_columns] + [f"property_{n}" for n in m.requires_properties]
            for i, m in enumerate(self.metrics)
        }

        unresovled_dependencies = {
            k: v for k, vv in self._dependencies.items() for v in vv if v not in self._dependencies
        }
        if len(unresovled_dependencies) > 0:
            error_dict = {f"self.metrics[{k}]": v for k, v in unresovled_dependencies.items()}
            raise RuntimeError(
                f"There are columns and properties required by metrics, that are never computed: {error_dict}"
            )

        self._parameters = [v for m in self.metrics for v in m._parameters]

        self.exclude_columns += [v for m in self.metrics for v in m.computes_intermediate_columns]
        self.exclude_properties += [v for m in self.metrics for v in m.computes_intermediate_properties]

        ts = graphlib.TopologicalSorter(self._dependencies)
        self._ordered_metrics = [self.metrics[o] for o in ts.static_order() if isinstance(o, int)]

    def __repr__(self):
        return f"computes columns: {[c for m in self._ordered_metrics for c in m.computes_columns]} - computes properties {[p for m in self._ordered_metrics for p in m.computes_properties]} - args {self._parameters}"

    def compute(self, r: Recording, **kwargs) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
        if "polygon" not in r._df.columns:
            r._df = r._add_polygons(r._df)
        if "geometry" not in r._df.columns:
            r._df = r._df.with_columns(geometry=st.from_shapely("polygon"))

        df = pl.LazyFrame(r._df)
        properties = {}
        for m in self._ordered_metrics:
            df, new_p = m.compute_lazy(
                df=df,
                **{k: properties[k] for k in m.requires_properties},
                **{k: v for k, v in kwargs.items() if k in [p.name for p in m._parameters]},
            )
            properties |= new_p
        for k in self.exclude_properties:
            del properties[k]
        df = df.drop(self.exclude_columns)
        res = pl.collect_all([df] + list(properties.values()))
        df, computed_props = res[0], res[1:]
        assert all(c in df.columns or c in self.exclude_columns for m in self.metrics for c in m.computes_columns)
        return df, {k: v for k, v in zip(properties.keys(), computed_props)}
