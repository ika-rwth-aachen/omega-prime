import polars as pl
from dataclasses import dataclass, field
from collections.abc import Callable
import polars_st as st
from .recording import Recording
import graphlib


@dataclass
class Metric:
    compute_func: Callable[[pl.LazyFrame, ...], tuple[pl.LazyFrame, dict[str, pl.LazyFrame]]]
    computes_columns: list[str] = field(default_factory=list)
    computes_properties: list[str] = field(default_factory=list)
    requires_columns: list[str] = field(default_factory=list)
    requires_properties: list[str] = field(default_factory=list)

    def compute_lazy(self, df, **kwargs) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
        return self.compute_func(df, **kwargs)


@dataclass
class MetricManager:
    metrics: list[Metric]
    _dependencies: dict[int | str, list[int | str]] = field(init=False)
    _ordered_metrics: list[Metric] = field(init=False)

    def __post_init__(self):
        self._dependencies = {
            val: [i]
            for i, m in enumerate(self.metrics)
            for val in [f"column_{n}" for n in m.computes_columns] + [f"property_{n}" for n in m.computes_properties]
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

        ts = graphlib.TopologicalSorter(self._dependencies)
        self._ordered_metrics = [self.metrics[o] for o in ts.static_order() if isinstance(o, int)]

    def __repr__(self):
        return f"computes columns: {[c for m in self._ordered_metrics for c in m.computes_columns]} - computes properties {[p for m in self._ordered_metrics for p in m.computes_properties]}"

    def compute(self, r: Recording, *args, **kwargs) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
        if "polygon" not in r._df.columns:
            r._df = r._add_polygons(r._df)
        if "geometry" not in r._df.columns:
            r._df = r._df.with_columns(geometry=st.from_shapely("polygon"))

        df = pl.LazyFrame(r._df)
        properties = {}
        for m in self._ordered_metrics:
            df, new_p = m.compute_lazy(df, *args, **{k: properties[k] for k in m.requires_properties}, **kwargs)
            properties |= new_p
        res = pl.collect_all([df] + list(properties.values()))
        df, computed_props = res[0], res[1:]
        return df, {k: v for k, v in zip(properties.keys(), computed_props)}


def add_driven_distance_and_vel(df, *args, **kwargs) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    return df.with_columns(
        (pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2)
        .sqrt()
        .over("idx")
        .fill_null(0.0)
        .cum_sum()
        .alias("distance_traveled"),
        (pl.col("vel_x") ** 2 + pl.col("vel_y") ** 2).sqrt().alias("vel"),
    ), {}


drivenDistancenAndVel = Metric(computes_columns=["distance_traveled", "vel"], compute_func=add_driven_distance_and_vel)


def get_timegaps(df, ego_id, *args, time_buffer=2e9, **kwargs):
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
        "timegaps": timegaps,
        "min_timegaps": min_timegaps,
        "p_timegaps": p_timegaps,
        "min_p_timegaps": min_p_timegaps,
    }


timegaps_and_p_timegaps = Metric(
    requires_columns=["distance_traveled", "vel"],
    compute_func=get_timegaps,
    computes_columns=[],
    computes_properties=["timegaps", "min_timegaps", "p_timegaps", "min_p_timegaps"],
)
