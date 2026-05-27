from enum import Enum
from abc import ABC, abstractmethod
from collections import namedtuple as nt
from pathlib import Path
from typing import Any
import shapely
import numpy as np
from matplotlib import pyplot as plt

from .locator import Locator


def _save_or_show(output_plot, filename: str):
    """Save figure to *output_plot* directory/file, or show it if *output_plot* is None."""
    try:
        if output_plot is None:
            plt.show()
        else:
            output_path = Path(output_plot)
            if output_path.is_dir() or not output_path.suffix:
                output_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path / filename)
            elif output_path.suffix in (".pdf", ".png", ".svg"):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path)
            else:
                raise ValueError(
                    f"output_plot must be a directory or a file path (.pdf/.png/.svg), got: {output_plot}"
                )
    finally:
        plt.close()


class MapSegmentType(Enum):
    """Classification of MapSegments."""

    STRAIGHT = "straight"
    JUNCTION = "junction"
    ROUNDABOUT = "roundabout"
    RAMP_ON = "ramp_on"
    RAMP_OFF = "ramp_off"
    UNKNOWN = "unknown"


class Segment(ABC):
    """A class that represents a segment of the map"""

    def __init__(self, lanes, idx=None, concave_hull_ratio=0.3):
        self.lanes = lanes
        self.lane_ids = [self._get_lane_id(lane) for lane in lanes]
        self.trafficlights = []
        self.idx = idx
        self.concave_hull_ratio = concave_hull_ratio
        self.type = MapSegmentType.UNKNOWN

        # Cache polygon to avoid recomputing concave hull when lanes stay unchanged
        self._polygon_cache = None
        self._polygon_cache_key = None
        self._polygon_dirty = True
        self.polygon = self.create_segment_polygon()

    @abstractmethod
    def _get_lane_id(self, lane):
        """Extract lane ID from a lane object. Map-type specific."""
        pass

    @abstractmethod
    def _get_lane_geometry(self, lane) -> shapely.LineString:
        """Extract geometry from a lane object. Map-type specific."""
        pass

    @abstractmethod
    def set_trafficlight(self):
        """Set traffic lights for this segment. Map-type specific."""
        pass

    def _compute_polygon_key(self):
        return tuple((self._get_lane_id(lane), self._get_lane_geometry(lane).wkb) for lane in self.lanes)

    def _compute_segment_polygon(self):
        lane_centerline = [self._get_lane_geometry(lane) for lane in self.lanes]

        multi_centerline = shapely.geometry.MultiLineString(lane_centerline)
        combined = multi_centerline.buffer(0.1)
        combined = combined.simplify(0.1, preserve_topology=True)

        try:
            hull = shapely.concave_hull(combined, self.concave_hull_ratio)
            assert not hull.is_empty
        except (shapely.errors.GEOSException, AssertionError):
            hull = shapely.convex_hull(combined)
            assert not hull.is_empty
        return hull

    def _ensure_polygon(self, force=False):
        key = self._compute_polygon_key()
        if force or self._polygon_dirty or key != self._polygon_cache_key:
            self._polygon_cache = self._compute_segment_polygon()
            self._polygon_cache_key = key
            self._polygon_dirty = False
        return self._polygon_cache

    def get_center_point(self):
        "Returns the center point of the segment"
        return self.polygon.centroid.x, self.polygon.centroid.y

    def create_segment_polygon(self):
        "Create the Polygon of the Segment"
        return self._ensure_polygon()

    def update_polygon(self):
        "Updates the Polygon of the Segment"
        self._polygon_dirty = True
        self.polygon = self._ensure_polygon(force=True)

    def add_lane(self, lanes, update_polygon=True):
        """Adds a lane to the segment.
        If the lane is already in the segment, it will not be added again.

        Args:
            lane (list): A list of lane objects to be added to the segment.
        """
        for lane in lanes:
            if lane not in self.lanes:
                self.lanes.append(lane)
                self.lane_ids.append(self._get_lane_id(lane))

        if update_polygon:
            self.update_polygon()

        self.set_trafficlight()

    def get_timeinterval_on_segment(self, roaduser):
        """
        Gets a roadsegment as input as well as a roaduser trajectory.
        Returns the time interval of the roaduser on the segment.
        roaduser should be a np.array with (total_nanos, x, y)
        """
        if self.polygon:
            roaduser_points = [shapely.Point(x, y) for x, y in roaduser[:, 1:3]]
            roaduser_on_segment = np.array([self.polygon.contains(point) for point in roaduser_points])
            if roaduser_on_segment.any():
                indices = np.where(roaduser_on_segment)[0]
                return roaduser[indices[0], 0], roaduser[indices[-1], 0]
            else:
                return None
        else:
            return None

    def _plot_lane_label(self, lane):
        return str(self._get_lane_id(lane))

    def _plot_filename_prefix(self):
        return self.__class__.__name__

    def _plot_color(self):
        if self.type == MapSegmentType.JUNCTION:
            return "green"
        if self.type == MapSegmentType.STRAIGHT:
            return "steelblue"
        return "blue"

    def _plot_title(self):
        if self.type == MapSegmentType.JUNCTION:
            segment_type = "Intersection"
        elif self.type == MapSegmentType.STRAIGHT:
            segment_type = "Connection"
        else:
            segment_type = "Segment"
        return f"{segment_type} {self.idx} ({len(self.lanes)} lanes)"

    def plot(self, output_plot: Path = None):
        """Plot this segment with its lane centerlines, lane labels, and polygon."""
        _, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        color = self._plot_color()

        for lane in self.lanes:
            geometry = self._get_lane_geometry(lane)
            ax.plot(*np.asarray(geometry.xy)[:2], color=color)

        for lane in self.lanes:
            geometry = self._get_lane_geometry(lane)
            m = int(np.ceil(len(geometry.xy[0]) / 2))
            ax.annotate(
                self._plot_lane_label(lane),
                xy=(geometry.xy[0][m], geometry.xy[1][m]),
                fontsize=2,
                color="black",
                zorder=3,
            )

        try:
            ax.fill(*self.polygon.exterior.xy, color=color, alpha=0.15, zorder=5)
            ax.plot(*self.polygon.exterior.xy, color=color, alpha=0.7, zorder=10)
        except Exception:
            pass

        plt.title(self._plot_title())
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        _save_or_show(output_plot, f"{self._plot_filename_prefix()}{self.idx}.pdf")


class MapSegmentation(ABC):
    """
    Abstract base class for map segmentation that handles multiple segments on a single map.
    Concrete implementations must define how to extract lane-specific information.
    """

    def __init__(self, recording, concave_hull_ratio=0.3):
        self.map = recording.map
        self.lanes = recording.map.lanes
        self.trafficlight = {}
        self.trafficlight_ids = set()
        self.intersections = []
        self.lane_dict = {}
        self.lane_successors_dict = {}
        self.lane_predecessors_dict = {}
        self.intersecting_lanes_dict = {}
        self.intersection_dict = {}
        self.lane_segment_dict = {}
        self.segments = []
        self.segment_by_road_id = {}
        self.concave_hull_ratio = concave_hull_ratio
        self.locator = Locator.from_map(recording.map)

        segment_name = nt("SegmentName", ["lane_id", "segment_idx", "segment"])
        for lane in self.lanes.values():
            self.lane_segment_dict[self._get_lane_id(lane)] = segment_name(self._get_lane_id(lane), None, None)

    @abstractmethod
    def _get_lane_id(self, lane) -> Any:
        """Extract lane ID from a lane object. Map-type specific."""
        pass

    @abstractmethod
    def _get_lane_centerline(self, lane) -> shapely.LineString:
        """Extract centerline from a lane object. Map-type specific."""
        pass

    @abstractmethod
    def _get_lane_successors(self, lane) -> list:
        """Extract successor IDs from a lane object. Map-type specific."""
        pass

    @abstractmethod
    def _get_lane_predecessors(self, lane) -> list:
        """Extract predecessor IDs from a lane object. Map-type specific."""
        pass

    @abstractmethod
    def _has_traffic_light(self, lane) -> bool:
        """Check if lane has traffic light. Map-type specific."""
        pass

    @abstractmethod
    def _get_traffic_light(self, lane):
        """Get traffic light object from lane. Map-type specific."""
        pass

    @abstractmethod
    def _set_lane_on_intersection(self, lane, value: bool):
        """Set the on_intersection attribute for a lane. Map-type specific."""
        pass

    @abstractmethod
    def _set_lane_is_approaching(self, lane, value: bool):
        """Set the is_approaching attribute for a lane. Map-type specific."""
        pass

    @abstractmethod
    def _get_lane_on_intersection(self, lane) -> bool:
        """Get the on_intersection status of a lane. Map-type specific."""
        pass

    def _get_lane_is_approaching(self, lane) -> bool:
        """Get the approaching status of a lane."""
        return getattr(lane, "is_approaching", False)

    def _plot_lane_label(self, lane):
        return str(self._get_lane_id(lane))

    def _plot_map_title(self):
        return "Map Segmentation"

    def _plot_map_filename(self):
        return "Map_Segmentation.pdf"

    def _plot_lane_color(self, lane):
        if self._get_lane_on_intersection(lane):
            return "green"
        if self._get_lane_is_approaching(lane):
            return "orange"
        return "black"

    def _plot_legend_handles(self):
        return [
            plt.Line2D([0], [0], color="green", label="Intersection lane"),
            plt.Line2D([0], [0], color="orange", label="Approaching lane"),
            plt.Line2D([0], [0], color="black", label="Road lane"),
        ]

    # Concrete methods using abstract methods
    def create_lane_dict(self):
        """Returns a dictionary mapping each lane's lane_id to the lane object."""
        self.lane_dict = {self._get_lane_id(lane): lane for lane in self.lanes.values()}
        return self.lane_dict

    def get_lane_successors_and_predecessors(self):
        """Returns dictionaries mapping each lane's lane_id to its successor and predecessor lane indices."""
        lane_successors = {}
        lane_predecessors = {}

        for lane in self.lanes.values():
            lane_id = self._get_lane_id(lane)
            lane_successors[lane_id] = self._get_lane_successors(lane)
            lane_predecessors[lane_id] = self._get_lane_predecessors(lane)

        self.lane_successors_dict = lane_successors
        self.lane_predecessors_dict = lane_predecessors
        return lane_successors, lane_predecessors

    def check_if_all_lanes_are_on_segment(self):
        """
        Checks if all lanes are on a segment.
        Returns:
            bool: True if all lanes are on a segment, False otherwise.
        """
        for lane in self.lanes.values():
            lane_id = self._get_lane_id(lane)
            if lane_id not in self.lane_segment_dict or self.lane_segment_dict[lane_id].segment is None:
                return False
        return True

    def _located_lane_id_to_segment_lane_id(self, located_lane_id):
        """Convert a locator lane ID to the key used by ``lane_segment_dict``."""
        return located_lane_id

    def _get_located_lane(self, located_lane_id, segment_lane_id):
        lane = self.lanes.get(located_lane_id)
        if lane is not None:
            return lane

        lane = self.lane_dict.get(segment_lane_id)
        if lane is not None:
            return lane

        for candidate in self.lanes.values():
            if self._get_lane_id(candidate) == segment_lane_id:
                return candidate
        return None

    def trajectory_segment_detection(self, trajectory):
        """Split a trajectory into chunks by the map segment of the located lane.

        Args:
            trajectory: ``np.ndarray`` with shape ``(n, 3)`` and columns
                ``(frame, x, y)``.

        Returns:
            A list of ``(trajectory_chunk, segment)`` tuples. Each chunk contains
            the original ``(frame, x, y)`` rows for one continuous segment.
        """
        if not isinstance(trajectory, np.ndarray):
            raise ValueError("trajectory must be a numpy array with shape (n, 3)")
        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError("trajectory must be a numpy array with shape (n, 3)")
        if trajectory.shape[0] == 0:
            return []

        sts = self.locator.xys2sts(trajectory[:, 1:3])
        located_lane_ids = sts["roadlane_id"].to_numpy()

        resolved_segments = []
        for i, located_lane_id in enumerate(located_lane_ids):
            segment_lane_id = self._located_lane_id_to_segment_lane_id(located_lane_id)
            entry = self.lane_segment_dict.get(segment_lane_id)
            if entry is None:
                raise RuntimeError(
                    "Could not assign trajectory point to a segment: "
                    f"point index {i}, frame {trajectory[i, 0]}, located lane {located_lane_id!r} "
                    f"is not present in lane_segment_dict"
                )
            if entry.segment is None:
                raise RuntimeError(
                    "Could not assign trajectory point to a segment: "
                    f"point index {i}, frame {trajectory[i, 0]}, located lane {located_lane_id!r} "
                    "has no assigned segment"
                )

            lane = self._get_located_lane(located_lane_id, segment_lane_id)
            if lane is None:
                raise RuntimeError(
                    "Could not assign trajectory point to a segment: "
                    f"point index {i}, frame {trajectory[i, 0]}, located lane {located_lane_id!r} "
                    "could not be resolved to a lane object"
                )

            polygon = getattr(lane, "polygon", None)
            if polygon is not None:
                point = shapely.Point(trajectory[i, 1], trajectory[i, 2])
                if not polygon.covers(point):
                    raise RuntimeError(
                        "Could not assign trajectory point to a segment: "
                        f"point index {i}, frame {trajectory[i, 0]}, point ({trajectory[i, 1]}, {trajectory[i, 2]}) "
                        f"is outside located lane {located_lane_id!r}"
                    )

            resolved_segments.append(entry.segment)

        segments = []
        current_segment = resolved_segments[0]
        current_rows = [trajectory[0]]
        for row, segment in zip(trajectory[1:], resolved_segments[1:]):
            if segment is current_segment:
                current_rows.append(row)
            else:
                segments.append((np.array(current_rows), current_segment))
                current_segment = segment
                current_rows = [row]

        segments.append((np.array(current_rows), current_segment))
        return segments

    def build_segment_by_road_id(self):
        """Build a mapping from road_id (as str) to the owning Segment.

        Works for both OSI (where update_road_ids() has aligned lane.idx.road_id
        with the segment index) and ODR (where lane.idx.road_id is the original
        ODR road ID string).
        """
        self.segment_by_road_id = {}
        for segment in self.segments:
            for lane in segment.lanes:
                self.segment_by_road_id[str(lane.idx.road_id)] = segment

    def get_segment(self, road_id) -> "Segment | None":
        """Return the Segment for a given road_id, or None if not found."""
        return self.segment_by_road_id.get(str(road_id))

    def plot(
        self,
        output_plot: Path = None,
        trajectory=None,
        plot_lane_ids=False,
        plot_intersection_polygons=False,
        plot_connection_polygons=False,
    ):
        """Plot the segmented map with lane colors, optional polygons, and optional trajectory."""
        _, ax = plt.subplots(1, 1)
        ax.set_aspect(1)

        for lane in self.lanes.values():
            ax.plot(
                *self._get_lane_centerline(lane).xy,
                color=self._plot_lane_color(lane),
                alpha=0.4,
                zorder=-10,
            )

        if plot_lane_ids:
            for lane in self.lanes.values():
                midpoint = self._get_lane_centerline(lane).interpolate(0.5, normalized=True)
                ax.annotate(self._plot_lane_label(lane), xy=(midpoint.x, midpoint.y), fontsize=2, color="black")

        for intersection in self.intersections:
            ax.annotate(
                intersection.idx,
                xy=intersection.get_center_point(),
                fontsize=4,
                color="darkgreen",
                fontweight="bold",
            )
            if plot_intersection_polygons:
                self._plot_segment_polygon(ax, intersection, "green", 0.15, 0.7)

        for connection in getattr(self, "isolated_connections", []):
            ax.annotate(connection.idx, xy=connection.get_center_point(), fontsize=4, color="steelblue")
            if plot_connection_polygons:
                self._plot_segment_polygon(ax, connection, "steelblue", 0.1, 0.5)

        self._plot_traffic_lights(ax)

        if trajectory is not None:
            ax.plot(trajectory[:, 1], trajectory[:, 2], color="yellow", alpha=0.8, linewidth=2, label="Trajectory")
            ax.plot(trajectory[0, 1], trajectory[0, 2], "go", markersize=8, label="Start")
            ax.plot(trajectory[-1, 1], trajectory[-1, 2], "ro", markersize=8, label="End")

        plt.title(self._plot_map_title())
        plt.xlabel("X Coordinate (m)", fontsize=10)
        plt.ylabel("Y Coordinate (m)", fontsize=10)
        plt.legend(handles=self._plot_legend_handles(), fontsize=7)
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        _save_or_show(output_plot, self._plot_map_filename())

    def _plot_segment_polygon(self, ax, segment, color, fill_alpha, line_alpha):
        try:
            segment.update_polygon()
            ax.fill(*segment.polygon.exterior.xy, color=color, alpha=fill_alpha, zorder=5)
            ax.plot(*segment.polygon.exterior.xy, color=color, alpha=line_alpha, zorder=10, linewidth=1)
        except Exception:
            pass

    def _plot_traffic_lights(self, ax):
        for traffic_light in getattr(self, "trafficlight", {}).values():
            position = getattr(getattr(traffic_light, "base", None), "position", None)
            if position is None:
                continue
            ax.plot(
                position.x,
                position.y,
                marker="o",
                color="red",
                markersize=2,
                label=f"Traffic Light {traffic_light.id}",
            )

    def plot_intersections(self, output_plot: Path = None):
        """Plot each intersection and isolated connection segment separately."""
        for intersection in self.intersections:
            intersection.plot(output_plot)
        for connection in getattr(self, "isolated_connections", []):
            connection.plot(output_plot)
