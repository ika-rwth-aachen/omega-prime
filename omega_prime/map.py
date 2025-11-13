import logging
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
from pathlib import Path
from warnings import warn
from tqdm.auto import tqdm
import json
from . import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OsiLaneId = namedtuple("OsiLaneId", ["road_id", "lane_id"])


def split_linestring(line, max_length):
    """
    Split a LineString into segments of maximum length.

    Args:
        line: shapely LineString to split
        max_length: Maximum length of each segment

    Returns:
        List of LineString segments
    """
    segments = []

    # If line is already short enough, return it as is
    if line.length <= max_length:
        return [line]

    # Number of segments needed
    n_segments = int(np.ceil(line.length / max_length))

    # Get evenly spaced points along the line
    points = [line.interpolate(i / n_segments, normalized=True) for i in range(n_segments + 1)]

    # Create line segments
    for i in range(n_segments):
        segment_coords = [points[i].coords[0], points[i + 1].coords[0]]
        segments.append(shapely.LineString(segment_coords))

    return segments


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
    type: types.LaneBoundaryType
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
    is_approaching: bool = field(init=False, default=None)

    @property
    def on_intersection(self):
        return self.type == betterosi.LaneClassificationType.TYPE_INTERSECTION

    @on_intersection.setter
    def on_intersection(self, value: bool):
        if value:
            self.type = betterosi.LaneClassificationType.TYPE_INTERSECTION
        else:
            self.type = betterosi.LaneClassificationType.DRIVING  # TODO: choose a better default?

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
    """Base class for Map representations"""

    lane_boundaries: dict[Any, LaneBoundary]
    lanes: dict[Any:Lane]

    _supported_file_suffixes = [".osi", ".mcap"]
    _binary_json_identifier = b"osi"

    def plot(self, ax: plt.Axes | None = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_aspect(1)
        for l in self.lanes.values():
            l.plot(ax)
        for b in self.lane_boundaries.values():
            b.plot(ax)

    @classmethod
    def from_file(cls, filepath, parse_map=True, **kwargs):
        "Create a Map instance from a file."
        first_gt = next(betterosi.read(filepath, return_ground_truth=True, mcap_return_betterosi=True))
        return cls.create(first_gt, **kwargs)

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
                    pl.Series(name="subtype", values=[o.subtype.name for o in self.lanes.values()]),
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
                tooltip=[
                    "properties.idx:N",
                    "properties.type:O",
                    "properties.subtype:O",
                    "properties.on_intersection:O",
                ],
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

    def map_to_centerline_mcap(self, output_mcap_path: Path = None) -> betterosi.GroundTruth:
        """
        Convert an Map to a MapOsiCenterline and save it as an MCAP file if the output path is provided.
        It returns the generated GroundTruth object from the generated MapOsiCenterline.

        Args:
            output_mcap_path: Path where the MCAP file will be saved
        Returns:
            betterosi.GroundTruth: The generated GroundTruth object
        """

        # Create a mapping from XodrLaneId to a simple integer ID
        lane_id_mapping = {}
        for idx, lane_idx in enumerate(self.lanes.keys()):
            lane_id_mapping[lane_idx] = idx

        # Create betterosi.Lane objects for each lane
        osi_lanes = []
        for lane in self.lanes.values():
            if not lane.centerline.is_valid or lane.centerline.is_empty:
                logging.warning(f"Warning: Skipping invalid lane {lane.idx}")
                continue

            # Check for NaN/inf coordinates
            coords = np.array(lane.centerline.coords)
            if not np.isfinite(coords).all():
                logging.warning(f"Warning: Lane {lane.idx} has non-finite coordinates, skipping")
                continue

            if len(coords) < 2:
                logging.warning(f"Warning: Lane {lane.idx} has insufficient points, skipping")
                continue
            # Get centerline coordinates
            centerline_coords = list(shapely.simplify(lane.centerline, 0.1).coords)
            if not len(centerline_coords) > 1:
                centerline_coords = list(lane.centerline.coords)
                if not len(centerline_coords) > 1:
                    # skip lanes with insufficient centerline points
                    logging.warning(f"Warning: Skipping lane {lane.idx} due to insufficient centerline points")
                    continue

            centerline = [betterosi.Vector3D(x=float(x), y=float(y), z=0.0) for x, y in centerline_coords]

            assert len(centerline_coords) > 1
            # Create lane pairing for successor/predecessor relationships
            lane_pairings = []

            # Get all unique combinations of predecessors and successors
            predecessors = [pred_id for pred_id in lane.predecessor_ids if pred_id in lane_id_mapping]
            successors = [succ_id for succ_id in lane.successor_ids if succ_id in lane_id_mapping]

            # If there are no predecessors or successors, create a single pairing with None values
            if predecessors or successors:
                # Create pairings for all combinations
                if not predecessors:
                    predecessors = [None]
                if not successors:
                    successors = [None]

                for pred_id in predecessors:
                    for succ_id in successors:
                        lane_pairings.append(
                            betterosi.LaneClassificationLanePairing(
                                antecessor_lane_id=betterosi.Identifier(value=lane_id_mapping[pred_id])
                                if pred_id is not None
                                else None,
                                successor_lane_id=betterosi.Identifier(value=lane_id_mapping[succ_id])
                                if succ_id is not None
                                else None,
                            )
                        )

            # Create the OSI lane
            osi_lane = betterosi.Lane(
                id=betterosi.Identifier(value=lane_id_mapping[lane.idx]),
                classification=betterosi.LaneClassification(
                    centerline=centerline,
                    centerline_is_driving_direction=True,
                    type=lane.type,
                    subtype=lane.subtype,
                    lane_pairing=lane_pairings,
                ),
            )
            osi_lanes.append(osi_lane)

        # Create a GroundTruth with only the lanes (no moving objects, no lane boundaries)
        ground_truth = betterosi.GroundTruth(
            version=betterosi.InterfaceVersion(
                version_major=3,
                version_minor=7,
                version_patch=0,
            ),
            timestamp=betterosi.Timestamp(
                seconds=0,
                nanos=0,
            ),
            lane=osi_lanes,
        )

        # Save to MCAP file if output path is provided
        if output_mcap_path is None:
            logging.warning("No output path provided for MCAP file")
        else:
            # Convert string to Path if needed
            output_mcap_path = Path(output_mcap_path)

            if output_mcap_path.is_dir():
                output_mcap_path = output_mcap_path / "map_to_centerline.mcap"
            elif not output_mcap_path.suffix == ".mcap":
                logging.warning(f"Output path must be a directory or .mcap file: {output_mcap_path}")
                return ground_truth

            with betterosi.Writer(output_mcap_path) as writer:
                writer.add(ground_truth, topic="ground_truth_map", log_time=0)
            logging.info(f"Successfully saved map with {len(osi_lanes)} lanes to {output_mcap_path}")

        return ground_truth

    def align_predecessor_and_successor_relations(self):
        """
        Ensure that predecessor and successor relationships between lanes are consistent.
        If lane A lists lane B as a successor, then lane B should list lane A as a predecessor, and vice versa.
        """
        for lane in self.lanes.values():
            for succ_id in lane.successor_ids:
                if succ_id in self.lanes:
                    succ_lane = self.lanes[succ_id]
                    if lane.idx not in succ_lane.predecessor_ids:
                        succ_lane.predecessor_ids.append(lane.idx)
            for pred_id in lane.predecessor_ids:
                if pred_id in self.lanes:
                    pred_lane = self.lanes[pred_id]
                    if lane.idx not in pred_lane.successor_ids:
                        pred_lane.successor_ids.append(lane.idx)

    @classmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError()

    def __post_init__(self):
        self.setup_lanes_and_boundaries()

    def setup_lanes_and_boundaries(self):
        raise NotImplementedError()

    def _to_binary_json(self):
        raise NotImplementedError()

    @classmethod
    def _from_binary_json(cls, d, **kwargs):
        raise NotImplementedError()


@dataclass(repr=False)
class MapOsi(Map):
    "Map representation based on ASAM OSI GroundTruth"

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

    def _to_binary_json(self):
        d = json.loads(self._osi.to_json())
        if "movingObject" in d:
            del d["movingObject"]
        return {b"osi": json.dumps(d).encode()}

    @classmethod
    def _from_binary_json(cls, d, **kwargs):
        gt = betterosi.GroundTruth().from_json(d[b"osi"].decode())
        if len(gt.lane_boundary) > 0:
            return cls.create(gt)
        else:
            return None


@dataclass(repr=False)
class MapOsiCenterline(Map):
    "Map representation based on ASAM OSI GroundTruth defining only the centerlines of lanes and nothing else. Does not conform to the omega-prime specification for Map."

    _osi: betterosi.GroundTruth
    lanes: dict[int, LaneOsiCenterline]

    @classmethod
    def create(cls, gt: betterosi.GroundTruth, split_lanes: bool = False, split_lanes_length: float = 10, **kwargs):
        if len(gt.lane) == 0:
            raise RuntimeError("No Map")
        c = cls(
            _osi=gt,
            lanes={l.idx: l for l in [LaneOsiCenterline.create(l) for l in gt.lane]},
            lane_boundaries={},
        )
        if split_lanes:
            c._split(split_lanes_length)
        return c

    def setup_lanes_and_boundaries(self):
        map_osi_id2idx = {l._osi.id.value: l.idx for l in self.lanes.values()}
        for l in self.lanes.values():
            l.successor_ids = [map_osi_id2idx[int(i)] for i in l.successor_ids if int(i) in map_osi_id2idx]
            l.predecessor_ids = [map_osi_id2idx[int(i)] for i in l.predecessor_ids if int(i) in map_osi_id2idx]
        for l in self.lanes.values():
            l._map = self

        # Sometimes a presuccessor lane is not set as a successor lane in the other lane, therefore we need to check where this is the case and add it
        self.align_predecessor_and_successor_relations()

    def _split(self, max_len: float):
        """
        Split lanes into segments of maximum length.

        This method post-processes the map by splitting lane centerlines that exceed
        the specified maximum length into smaller segments. It updates lane connections
        accordingly and removes connections between segments that are too far apart.

        Args:
            max_len (float): Maximum length allowed for each lane segment.
        """
        warn("The Postprocessing is ACTIVE! The lanes will be split into segments!!!")
        lanes_or = self.lanes
        lanes_new = {}
        idx_count = 0

        for lane in tqdm(lanes_or.values()):
            if lane.centerline.length > max_len:
                # Split the lane's centerline into segments of maximum length
                segments = split_linestring(lane.centerline, max_len)
            else:
                segments = [lane.centerline]

            # Create new lane objects for each segment
            segment_lanes = []
            for i, segment in enumerate(segments):
                # Create a copy of the lane with modified centerline
                # new_lane = copy.deepcopy(lane)

                new_lane = LaneOsiCenterline(
                    _osi=lane._osi,
                    idx=OsiLaneId(road_id=idx_count, lane_id=idx_count),
                    centerline=segment,
                    type=lane.type,
                    subtype=lane.subtype,
                    successor_ids=[],
                    predecessor_ids=[],
                )

                segment_lanes.append(new_lane)
                lanes_new[new_lane.idx.lane_id] = new_lane
                idx_count += 1

            for i, new_lane in enumerate(segment_lanes):
                if len(segments) == 1:
                    # If only one segment, keep original predecessors and successors
                    new_lane.predecessor_ids = lane.predecessor_ids
                    new_lane.successor_ids = lane.successor_ids
                elif i == 0:
                    # First segment: keep original predecessors, connect to next segment
                    new_lane.predecessor_ids = lane.predecessor_ids
                    new_lane.successor_ids = [segment_lanes[i + 1].idx]
                elif i == len(segments) - 1:
                    # Last segment: connect to previous segment, keep original successors
                    new_lane.predecessor_ids = [segment_lanes[i - 1].idx]
                    new_lane.successor_ids = lane.successor_ids
                else:
                    # Middle segments: connect to both neighbors
                    new_lane.predecessor_ids = [segment_lanes[i - 1].idx]
                    new_lane.successor_ids = [segment_lanes[i + 1].idx]

            # Update references in other lanes' predecessors/successors
            for other_lane in lanes_or.values():
                if lane.idx in other_lane.successor_ids:
                    # Replace reference to original lane with first segment
                    idx = other_lane.successor_ids.index(lane.idx)
                    other_lane.successor_ids[idx] = segment_lanes[0].idx
                if lane.idx in other_lane.predecessor_ids:
                    # Replace reference to original lane with last segment
                    idx = other_lane.predecessor_ids.index(lane.idx)
                    other_lane.predecessor_ids[idx] = segment_lanes[-1].idx

        # Replace original lanes with segmented lanes

        # Do a check for the predecessor and successor: Check if the distance between the centerlines is greater than the max_len --> if yes, then remove the connection
        for lane in lanes_new.values():
            if lane.predecessor_ids:
                for pre in lane.predecessor_ids:
                    pre_to_remove = []
                    if lanes_new[pre.lane_id].centerline.distance(lane.centerline) > max_len:
                        pre_to_remove.append(pre)
                        try:
                            lanes_new[pre.lane_id].successor_ids.remove(lane.idx)
                        except ValueError:
                            pass  # If the successor is not in the list, ignore

                    for pre in pre_to_remove:
                        try:
                            lanes_new[lane.idx.lane_id].predecessor_ids.remove(pre)
                        except ValueError:
                            pass  # If the predecessor is not in the list, ignore
            if lane.successor_ids:
                for suc in lane.successor_ids:
                    suc_to_remove = []
                    if lanes_new[suc.lane_id].centerline.distance(lane.centerline) > max_len:
                        suc_to_remove.append(suc)
                        try:
                            lanes_new[suc.lane_id].predecessor_ids.remove(lane.idx)
                        except ValueError:
                            pass  # If the predecessor is not in the list, ignore

                    for suc in suc_to_remove:
                        try:
                            lanes_new[lane.idx.lane_id].successor_ids.remove(suc)
                        except ValueError:
                            pass

        self.lanes = {lane.idx: lane for lane in lanes_new.values()}
        for lane in self.lanes.values():
            lane._map = self
        return self

    def _to_binary_json(self):
        d = json.loads(self._osi.to_json())
        if "movingObject" in d:
            del d["movingObject"]
        return {b"osi": json.dumps(d).encode()}

    @classmethod
    def _from_binary_json(cls, d, **kwargs):
        gt = betterosi.GroundTruth().from_json(d[b"osi"].decode())
        return cls.create(gt)
