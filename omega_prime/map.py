from dataclasses import dataclass, field
from typing import Any
from collections import namedtuple
import betterosi
import numpy as np
import shapely
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as PltPolygon
import polars as pl
import altair as alt
import polars_st as st


OsiLaneId = namedtuple("OsiLaneId", ["road_id", "lane_id"])


@dataclass
class ProjectionOffset:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0


@dataclass(repr=False)
class LaneBoundary:
    _map: "Map" = field(init=False)
    idx: Any
    type: betterosi.LaneBoundaryClassificationType
    polyline: shapely.LineString
    # reference: Any = field(init=False, default=None)

    def plot(self, ax: plt.Axes | None = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_aspect(1)
        ax.plot(*np.array(self.polyline.coords)[:, :2].T, color="gray", alpha=0.1)

    def get_osi(self) -> betterosi.LaneBoundary:
        raise NotImplementedError()

    @classmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError()


@dataclass(repr=False)
class LaneBoundaryOsi(LaneBoundary):
    _osi: betterosi.LaneBoundary

    @classmethod
    def create(cls, lane_boundary: betterosi.LaneBoundary):
        return cls(
            idx=lane_boundary.id.value,
            polyline=shapely.LineString([(p.position.x, p.position.y) for p in lane_boundary.boundary_line]),
            type=betterosi.LaneBoundaryClassificationType(lane_boundary.classification.type),
            _osi=lane_boundary,
        )

    def get_osi(self) -> betterosi.LaneBoundary:
        return self._osi


@dataclass(repr=False)
class LaneBase:
    _map: "Map" = field(init=False)
    idx: Any
    centerline: shapely.LineString
    type: betterosi.LaneClassificationType
    subtype: betterosi.LaneClassificationSubtype
    successor_ids: list[Any]
    predecessor_ids: list[Any]
    trafficlight: Any = field(init=False, default=None)

    @property
    def on_intersection(self):
        return self.type == betterosi.LaneClassificationType.TYPE_INTERSECTION

    def plot(self, ax: plt.Axes | None = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_aspect(1)
        c = "black" if not self.on_intersection else "green"
        ax.plot(*np.asarray(self.centerline.coords).T, color=c, alpha=0.3, zorder=-10)
        if hasattr(self, "polygon") and self.polygon is not None:
            if isinstance(self.polygon, shapely.MultiPolygon):
                ps = self.polygon.geoms
            else:
                ps = [self.polygon]
            for p in ps:
                ax.add_patch(PltPolygon(p.exterior.coords, fc="blue", alpha=0.2, ec=c))


@dataclass(repr=False)
class Lane(LaneBase):
    right_boundary_id: Any
    left_boundary_id: Any
    polygon: shapely.Polygon = field(init=False)
    left_boundary: LaneBoundary = field(init=False)
    right_boundary: LaneBoundary = field(init=False)
    _oriented_borders: Any = field(init=False, default=None)
    _start_points: Any = field(init=False, default=None)
    _end_points: Any = field(init=False, default=None)

    # for ase_engine/omega_prime
    def _get_oriented_borders(self):
        center_start = self.centerline.interpolate(0, normalized=True)
        left = self.left_boundary.polyline
        invert_left = left.project(center_start, normalized=True) > 0.5
        if invert_left:
            left = shapely.reverse(left)
        right = self.right_boundary.polyline
        invert_right = right.project(center_start, normalized=True) > 0.5
        if invert_right:
            right = shapely.reverse(right)
        return left, right

    @property
    def oriented_borders(self):
        if self._oriented_borders is None:
            self._oriented_borders = self._get_oriented_borders()
        return self._oriented_borders

    @property
    def start_points(self):
        if self._start_points is None:
            self._start_points = np.array([b.interpolate(0, normalized=True) for b in self.oriented_borders])
        return self._start_points

    @property
    def end_points(self):
        if self._end_points is None:
            self._end_points = np.array([b.interpolate(1, normalized=True) for b in self.oriented_borders])
        return self._end_points


@dataclass(repr=False)
class LaneOsiCenterline(LaneBase):
    _osi: betterosi.Lane
    left_boundary = None
    right_boundary = None

    @staticmethod
    def _get_centerline(lane: betterosi.Lane):
        cl = np.array([(p.x, p.y) for p in lane.classification.centerline])
        if not lane.classification.centerline_is_driving_direction:
            cl = np.flip(cl, axis=0)
        return shapely.LineString(cl)

    @classmethod
    def create(cls, lane: betterosi.Lane):
        successor_ids = [
            p.successor_lane_id.value for p in lane.classification.lane_pairing if p.successor_lane_id is not None
        ]
        predecessor_ids = [
            p.antecessor_lane_id.value for p in lane.classification.lane_pairing if p.antecessor_lane_id is not None
        ]
        lid = lane.id.value
        return cls(
            _osi=lane,
            idx=OsiLaneId(road_id=lid, lane_id=lid),
            centerline=cls._get_centerline(lane),
            type=betterosi.LaneClassificationType(lane.classification.type),
            subtype=betterosi.LaneClassificationSubtype(lane.classification.subtype),
            successor_ids=np.array(list(set(successor_ids))),
            predecessor_ids=np.array(list(set(predecessor_ids))),
        )


@dataclass(repr=False)
class LaneOsi(Lane, LaneOsiCenterline):
    right_boundary_ids: list[int]
    left_boundary_ids: list[int]
    free_boundary_ids: list[int]

    @classmethod
    def create(cls, lane: betterosi.Lane):
        lid = int(lane.id.value)
        return cls(
            _osi=lane,
            idx=OsiLaneId(road_id=lid, lane_id=lid),
            centerline=cls._get_centerline(lane),
            type=betterosi.LaneClassificationType(lane.classification.type),
            subtype=betterosi.LaneClassificationSubtype(lane.classification.subtype),
            successor_ids=[
                p.successor_lane_id.value for p in lane.classification.lane_pairing if p.successor_lane_id is not None
            ],
            predecessor_ids=[
                p.antecessor_lane_id.value for p in lane.classification.lane_pairing if p.antecessor_lane_id is not None
            ],
            right_boundary_ids=[idx.value for idx in lane.classification.right_lane_boundary_id if idx is not None],
            left_boundary_ids=[idx.value for idx in lane.classification.left_lane_boundary_id if idx is not None],
            right_boundary_id=[idx.value for idx in lane.classification.right_lane_boundary_id if idx is not None][0],
            left_boundary_id=[idx.value for idx in lane.classification.left_lane_boundary_id if idx is not None][0],
            free_boundary_ids=[idx.value for idx in lane.classification.free_lane_boundary_id if idx is not None],
        )

    def set_boundaries(self):
        self.left_boundary = self._map.lane_boundaries[self.left_boundary_ids[0]]
        self.right_boundary = self._map.lane_boundaries[self.right_boundary_ids[0]]

        # for omega

    def set_polygon(self):
        self.polygon = shapely.Polygon(
            np.concatenate(
                [
                    np.array(self.left_boundary.polyline.coords),
                    np.flip(np.array(self.right_boundary.polyline.coords), axis=0),
                ]
            )
        )
        if not self.polygon.is_simple:
            self.polygon = shapely.convex_hull(self.polygon)
        # TODO: fix or warning


@dataclass(repr=False)
class Map:
    lane_boundaries: dict[Any, LaneBoundary]
    lanes: dict[Any:Lane]

    _supported_file_suffixes = [".osi", ".mcap"]

    def plot(self, ax: plt.Axes | None = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_aspect(1)
        for l in self.lanes.values():
            l.plot(ax)
        for b in self.lane_boundaries.values():
            b.plot(ax)

    @classmethod
    def from_file(cls, filepath, parse_map=True):
        first_gt = betterosi.read(filepath, return_ground_truth=True, mcap_return_betterosi=True)
        return cls.create(first_gt)

    def plot_altair(self, recording=None, plot_polys=True):
        arbitrary_lane = next(iter(self.lanes.values()))
        plot_polys = hasattr(arbitrary_lane, "polygon") and arbitrary_lane.polygon is not None and plot_polys

        if not hasattr(self, "_plot_dict"):
            if plot_polys:
                shapely_series = pl.Series(
                    name="shapely", values=[l.polygon.simplify(0.1) for l in self.lanes.values()]
                )
            else:
                shapely_series = pl.Series(
                    name="shapely", values=[l.centerline.simplify(0.1) for l in self.lanes.values()]
                )

            map_df = pl.DataFrame(
                [
                    shapely_series,
                    pl.Series(name="idx", values=[i for i, _ in enumerate(self.lanes.keys())]),
                    pl.Series(name="type", values=[o.type.name for o in self.lanes.values()]),
                    pl.Series(name="on_intersection", values=[o.on_intersection for o in self.lanes.values()]),
                ]
            )
            map_df = map_df.with_columns(geometry=st.from_shapely("shapely")).drop("shapely")

            if recording is not None:
                buffer = 5
                [xmin], [xmax], [ymin], [ymax] = recording._df.select(
                    (pl.col("x").min() - buffer).alias("xmin"),
                    (pl.col("x").max() + buffer).alias("xmax"),
                    (pl.col("y").min() - buffer).alias("ymin"),
                    (pl.col("y").max() + buffer).alias("ymax"),
                )[0]

                pov_df = pl.DataFrame(
                    {"polygon": [shapely.Polygon([[xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax]])]}
                )
                pov_df = pov_df.select(geometry=st.from_shapely("polygon"))
                map_df = map_df.with_columns(
                    pl.col("geometry").st.intersection(pl.lit(pov_df["geometry"])),
                )
            self._plot_dict = {"values": map_df.st.to_dicts()}

        c = (
            alt.Chart(self._plot_dict)
            .mark_geoshape(fillOpacity=0.4, filled=True if plot_polys else False)
            .encode(
                tooltip=["properties.idx:N", "properties.type:O", "properties.on_intersection:O"],
                color=(
                    alt.when(alt.FieldEqualPredicate(equal=True, field="properties.on_intersection"))
                    .then(alt.value("black"))
                    .otherwise(alt.value("green"))
                ),
            )
        )
        if recording is None:
            return c.properties(title="Map").project("identity", reflectY=True)
        else:
            return c

    @classmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError()

    def __post_init__(self):
        self.setup_lanes_and_boundaries()

    def setup_lanes_and_boundaries(self):
        raise NotImplementedError()


@dataclass(repr=False)
class MapOsi(Map):
    _osi: betterosi.GroundTruth

    @classmethod
    def create(cls, gt: betterosi.GroundTruth):
        if len(gt.lane_boundary) == 0:
            raise RuntimeError("Empty Map")
        return cls(
            _osi=gt,
            lane_boundaries={b.id.value: LaneBoundaryOsi.create(b) for b in gt.lane_boundary},
            lanes={
                l.idx: l
                for l in [LaneOsi.create(l) for l in gt.lane if len(l.classification.right_lane_boundary_id) > 0]
            },
        )

    def __post_init__(self):
        self.setup_lanes_and_boundaries()

    def setup_lanes_and_boundaries(self):
        for b in self.lane_boundaries.values():
            b._map = self
        map_osi_id2idx = {l._osi.id.value: l.idx for l in self.lanes.values()}
        for l in self.lanes.values():
            l.successor_ids = [map_osi_id2idx[i] for i in l.successor_ids if i in map_osi_id2idx]
            l.predecessor_ids = [map_osi_id2idx[i] for i in l.predecessor_ids if i in map_osi_id2idx]
            l._map = self
            l.set_boundaries()
            l.set_polygon()


@dataclass(repr=False)
class MapOsiCenterline(Map):
    _osi: betterosi.GroundTruth
    lanes: dict[int, LaneOsiCenterline]

    @classmethod
    def create(cls, gt: betterosi.GroundTruth):
        if len(gt.lane) == 0:
            raise RuntimeError("No Map")
        c = cls(
            _osi=gt,
            lanes={l.idx: l for l in [LaneOsiCenterline.create(l) for l in gt.lane]},
            lane_boundaries={},
        )
        return c

    def setup_lanes_and_boundaries(self):
        map_osi_id2idx = {l._osi.id.value: l.idx for l in self.lanes.values()}
        for l in self.lanes.values():
            l.successor_ids = [map_osi_id2idx[int(i)] for i in l.successor_ids if int(i) in map_osi_id2idx]
            l.predecessor_ids = [map_osi_id2idx[int(i)] for i in l.predecessor_ids if int(i) in map_osi_id2idx]
        for l in self.lanes.values():
            l._map = self
