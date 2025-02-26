import betterosi
import shapely
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PltPolygon
from dataclasses import dataclass, field
from typing import Any

@dataclass(repr=False)
class LaneBoundary():
    _map: "Map" = field(init=False)
    idx: int
    type: betterosi.LaneBoundaryClassificationType
    polyline: shapely.LineString
    reference: Any = None
    
    def plot(self, ax: plt.Axes):
        ax.plot(*np.array(self.polyline.coords).T, color='gray', alpha=.1)
    
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
            _osi = lane_boundary
        )
        
    def get_osi(self) -> betterosi.LaneBoundary:
        return self._osi

@dataclass(repr=False)
class Lane():
    _map: "Map" = field(init=False)
    idx: int
    centerline: np.ndarray
    type: betterosi.LaneClassificationType
    subtype: betterosi.LaneClassificationSubtype
    successor_ids: list[int]
    predecessor_ids: list[int]
    right_boundary_id: int
    left_boundary_id: int
    polygon: shapely.Polygon = field(init=False)
    left_boundary: LaneBoundary = field(init=False)
    right_boundary: LaneBoundary = field(init=False)
    source_reference: Any = None
    

@dataclass(repr=False)
class LaneOsi(Lane):
    _osi: betterosi.Lane
    right_boundary_ids: list[int]
    left_boundary_ids: list[int]
    free_boundary_ids: list[int]
    
    
    @staticmethod
    def _get_centerline(lane:betterosi.Lane):
        cl = np.array([(p.x, p.y) for p in lane.classification.centerline])
        if not lane.classification.centerline_is_driving_direction:
            cl = np.flip(cl, axis=0)
        return cl

    @classmethod
    def create(cls, lane: betterosi.Lane):
        return cls(
            _osi = lane,
            idx=int(lane.id.value),
            centerline=cls._get_centerline(lane),
            type=betterosi.LaneClassificationType(lane.classification.type),
            subtype = betterosi.LaneClassificationSubtype(lane.classification.subtype),
            successor_ids = [p.successor_lane_id.value for p in lane.classification.lane_pairing if p.successor_lane_id is not None],
            predecessor_ids = [p.antecessor_lane_id.value for p in lane.classification.lane_pairing if p.antecessor_lane_id is not None],
            right_boundary_ids = [idx.value for idx in lane.classification.right_lane_boundary_id if idx is not None],
            left_boundary_ids = [idx.value for idx in lane.classification.left_lane_boundary_id if idx is not None],
            right_boundary_id = [idx.value for idx in lane.classification.right_lane_boundary_id if idx is not None][0],
            left_boundary_id = [idx.value for idx in lane.classification.left_lane_boundary_id if idx is not None][0],
            free_boundary_ids = [idx.value for idx in lane.classification.free_lane_boundary_id if idx is not None],

        )
        
    def set_boundaries(self):
        self.left_boundary = self._map.lane_boundaries[self.left_boundary_ids[0]]
        self.right_boundary = self._map.lane_boundaries[self.right_boundary_ids[0]]

        # for omega
        self.oriented_borders = self._get_oriented_borders()
        self.start_points = np.array([b.interpolate(0, normalized=True) for b in self.oriented_borders])
        self.end_points = np.array([b.interpolate(1, normalized=True) for b in self.oriented_borders])
        return self

    def set_polygon(self):
        self.polygon = shapely.Polygon(np.concatenate([np.array(self.left_boundary.polyline.coords), np.flip(np.array(self.right_boundary.polyline.coords), axis=0)]))
        if not self.polygon.is_simple:
            self.polygon = shapely.convex_hull(self.polygon)
        # TODO: fix or warning
            

    
    def plot(self, ax: plt.Axes):
        c = 'green' if not self.type==betterosi.LaneClassificationType.TYPE_INTERSECTION else 'black'
        ax.plot(*np.array(self.centerline).T, color=c, alpha=0.5)
        ax.add_patch(PltPolygon(self.polygon.exterior.coords, fc='blue', alpha=.2, ec='black'))
        
    # for ase_engine/omega_format

    
    def _get_oriented_borders(self):
        center_start = shapely.LineString(self.centerline).interpolate(0, normalized=True)
        left = self.left_boundary.polyline
        invert_left = left.project(center_start, normalized=True)>.5
        if invert_left:
            left = shapely.reverse(left)
        right = self.right_boundary.polyline
        invert_right = right.project(center_start, normalized=True)>.5
        if invert_right:
            right = shapely.reverse(right)
        return left, right

        

@dataclass(repr=False)
class Map():
    lane_boundaries: dict[int, LaneBoundary]
    lanes: dict[int: Lane]
    
    def plot(self, ax: plt.Axes):
        for l in self.lanes.values():
            l.plot(ax)
        for b in self.lane_boundaries.values():
            b.plot(ax)

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
        if not hasattr(gt, 'lane') or not hasattr(gt, 'lane_boundary'):
            return None
    
        return cls(
            _osi = gt,
            lane_boundaries = {b.id.value: LaneBoundaryOsi.create(b) for b in gt.lane_boundary},
            lanes = {l.id.value: LaneOsi.create(l) for l in gt.lane if len(l.classification.right_lane_boundary_id)>0}
        )
        
    def __post_init__(self):
        self.setup_lanes_and_boundaries()

            
    def setup_lanes_and_boundaries(self):
        for b in self.lane_boundaries.values():
            b._map = self
        for l in self.lanes.values():
            l._map = self
            l.set_boundaries()
            l.set_polygon()
        
