from enum import Enum
from abc import ABC, abstractmethod
from collections import namedtuple as nt
from typing import Any
import shapely
import numpy as np


class MapSegmentType(Enum):  # Added class for map segment from the scenario-database
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
        multilinestring = shapely.MultiLineString(lane_centerline).buffer(0.1)
        combined = shapely.unary_union(multilinestring).buffer(0.1)
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
        self.concave_hull_ratio = concave_hull_ratio

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
