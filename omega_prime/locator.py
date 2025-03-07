from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import betterosi
import networkx as nx
import numpy as np
import shapely
import xarray as xr
from matplotlib.patches import Polygon as PltPolygon
from strenum import StrEnum


class ShapelyTrajectoryTools:
    epsi = 1e-10
    l_append = 200
    simplify_tolerance=1e-5
    
    @staticmethod
    def plot_polygon(ax, p, **kwargs):
        if not p.is_empty:
            if isinstance(p, shapely.geometry.MultiPolygon):
                ps = p.geoms
            else:
                ps = [p]
            for p in ps:
                try:
                    ax.add_patch(PltPolygon(p.exterior.coords, **kwargs))
                except AttributeError as e:
                    raise e
                    # p is not a polygon
                    pass
    
    @classmethod
    def extend_linestring(cls, l: shapely.LineString, simplify:bool = True, l_append=None):
        l_append = l_append if l_append is not None else cls.l_append
        # TODO move s coordinate epsi to NOT normalized
        if isinstance(l, shapely.Point):
            cl = np.asarray(l.coords)[0]
            return shapely.LineString(np.stack([cl-np.array([0,1]), cl, cl+np.array([0,1])]))

        startvec = np.asarray(l.interpolate(0,normalized=True).coords)-np.asarray(l.interpolate(0+cls.epsi,normalized=True).coords)
        endvec = np.asarray(l.interpolate(1,normalized=True).coords)-np.asarray(l.interpolate(1-cls.epsi, normalized=True).coords)
        if np.all(startvec==0):
            startvec = np.asarray(l.interpolate(0,normalized=True).coords)-np.asarray(l.interpolate(1,normalized=True).coords)
        if np.all(endvec==0):
            endvec = np.asarray(l.interpolate(1,normalized=True).coords)-np.asarray(l.interpolate(0, normalized=True).coords)

        startvec = startvec/np.linalg.norm(startvec)*l_append
        endvec = endvec/np.linalg.norm(endvec)*l_append
        cl = np.asarray(l.coords)
        cl = shapely.LineString(np.concatenate([startvec+cl[0,:], cl, endvec+cl[-1,:]]))
        return cl.simplify(tolerance=cls.simplify_tolerance) if simplify else cl
        
    @classmethod
    def get_linestring_coordinate_s(cls, l: shapely.LineString):
        return shapely.line_locate_point(l, shapely.points(l.coords))
    
    @classmethod
    def st2xy(cls, l: shapely.LineString, s, t, return_heading_of_ref_at_st=False):
        ref_points_after = l.interpolate(np.clip(s+cls.epsi, 0, l.length))
        ref_points= l.interpolate(np.clip(s, 0, l.length))
        ref_points_before = l.interpolate(np.clip(s-cls.epsi, 0, l.length))
        lane_tangent_vec = np.array([np.array(o.coords)[0] for o in ref_points_after])-np.array([np.array(o.coords)[0] for o in ref_points_before])
        lane_tangent_norm_angle = np.arctan2(lane_tangent_vec[:,1],lane_tangent_vec[:,0])-np.pi/2
        if return_heading_of_ref_at_st:
            return lane_tangent_norm_angle+np.pi/2
        else:
            lane_tangent_complex_vec = np.exp(1j*lane_tangent_norm_angle)*t
            ref_points = np.array([np.array(o.coords)[0] for o in ref_points])
            x = np.real(lane_tangent_complex_vec)+ref_points[:,0]
            y = np.imag(lane_tangent_complex_vec)+ref_points[:,1]
            return x, y
    
    @classmethod
    def needs_angle_adjustment(cls, lane_point_distance,  lon_distances):
        lpd = lane_point_distance[:,np.newaxis]
        ub = lpd+cls.epsi
        lb = lpd-cls.epsi
        in_corner_area = np.logical_and(ub<=lon_distances, lb<=lon_distances)
        return np.any(in_corner_area, axis=0)
    
    @classmethod
    def xy2st(cls, l: shapely.LineString, x_or_xy, y=None, line_point_distances=None):
        if y is None:
            xy_points = x_or_xy
        else:
            xy_points = shapely.points(np.stack([x_or_xy,y],axis=1))
        lon_distances = l.project(xy_points)
        is_driver_side_of_centerline =np.asarray([o.is_ccw for o in shapely.linearrings(np.array([
            [np.asarray(o.coords)[0] for o in l.interpolate(np.clip(lon_distances - cls.epsi, 0, np.inf))],
            [np.asarray(o.coords)[0] for o in l.interpolate(np.clip(lon_distances + cls.epsi, 0, l.length))],
            [np.asarray(o.coords)[0] for o in xy_points]]).transpose(1,0,2))])
        lat_distances = l.distance(xy_points) * (-is_driver_side_of_centerline.astype(int)*2+1)

        delta_s = np.zeros_like(lon_distances)
        if line_point_distances is None:
            lane_point_distances = cls.get_linestring_coordinate_s(l)
        else:
            lane_point_distances = line_point_distances
        delta_s_idxs = cls.needs_angle_adjustment(lane_point_distances, lon_distances)
        if np.sum(delta_s_idxs)>0:
            points = np.stack([p.centroid.coords for p in xy_points[delta_s_idxs]])[:,0,:]
            lon_dist_to_fix = lon_distances[delta_s_idxs]
            before_nearest_points = np.stack([np.asarray(o.coords)[0] for o in l.interpolate(np.clip(lon_dist_to_fix - cls.epsi, 0, l.length))])
            nearest_points = np.stack([np.asarray(o.coords)[0] for o in l.interpolate(np.clip(lon_dist_to_fix, 0, l.length))])
            after_nearest_points = np.stack([np.asarray(o.coords)[0] for o in l.interpolate(np.clip(lon_dist_to_fix + cls.epsi, 0, l.length))])

            before_vec = nearest_points-before_nearest_points
            before_angle = np.arctan2(before_vec[:,1], before_vec[:,0])
            after_vec = after_nearest_points-nearest_points
            after_angle = np.arctan2(after_vec[:,1], after_vec[:,0])
            point_vec = points - nearest_points
            point_angle = np.arctan2(point_vec[:,1], point_vec[:,0])
            normal_angle = point_angle - np.pi/2
            before_in_angle = np.mod((before_angle - normal_angle),np.pi)
            after_in_angle = np.mod(normal_angle + np.pi - after_angle,np.pi)
            aia_sin = np.sin(after_in_angle)
            iz = np.isclose(aia_sin,0)
            gamma = np.ones_like(before_in_angle)*np.inf
            gamma[~iz] = np.sin(before_in_angle[~iz])/aia_sin[~iz]
            after_length = 2*cls.epsi*(1-(1/(gamma+1)))
            delta_s[delta_s_idxs] = cls.epsi-after_length
        return np.stack([lon_distances-delta_s, lat_distances]).T


def get_lane_centerline(right_border: shapely.LineString, left_border: shapely.LineString) -> shapely.LineString:
    """middle line between (interpolated) boundaries, oriented in direction of lane"""
    ses = np.unique(np.concatenate([shapely.line_locate_point(left_border, shapely.points(right_border.coords), normalized=True), shapely.line_locate_point(right_border, shapely.points(left_border.coords), normalized=True)]))
    
    points = np.zeros((len(ses),2))
    for i, (rbp, lbp) in enumerate(zip(right_border.interpolate(ses, normalized=True), left_border.interpolate(ses, normalized=True))):
        points[i, :] = shapely.MultiPoint([rbp, lbp]).minimum_rotated_rectangle.centroid.coords
    
    # TODO some smoothing operation could be helpful
    cl = shapely.LineString(points)

    if cl.is_empty or not cl.is_valid:
        raise RuntimeError("Could not compute centerline for lane!")
    return cl
  
class LaneRelation(StrEnum):
    predecessor = 'predecessor'
    successor = 'successor'
    neighbour_right = 'right neighbour'
    neighbour_left = 'left neighbour'
   
@dataclass
class MockRoad:
    odr_idx: int
    
@dataclass
class WrappedLane:
    idx: Any
    road: Any
    centerline: shapely.LineString
    polygon: shapely.Polygon
    border_left: shapely.LineString
    border_right: shapely.LineString
    predecessor_idxs: list[Any]
    successor_idxs: list[Any]
    start_points: list[shapely.Point]
    end_points: list[shapely.Point]
    on_intersection: bool = False
    type: Any = None
    original_lane: Any = None
    db: Any = None
    
    @classmethod
    def from_odr_lane(cls, map, l):
        start_points = np.stack([l.left_boundary.polyline[0,:2], l.right_boundary.polyline[0,:2]])
        end_points = np.stack([l.left_boundary.polyline[-1,:2], l.right_boundary.polyline[-1,:2]])
        l.odr_idx = l.idx[1]
        l.road = MockRoad(odr_idx=l.idx[0])
        return cls(
            idx=l.idx,
            road = l.road,
            centerline=shapely.LineString(l.centerline[:,:2]),
            polygon=l.polygon,
            border_left=shapely.LineString(l.left_boundary.polyline[:,:2]),
            border_right=shapely.LineString(l.right_boundary.polyline[:,:2]),
            predecessor_idxs=l.predecessor_ids,
            successor_idxs=l.successor_ids,
            start_points=start_points,
            end_points=end_points,
            on_intersection=l.type==betterosi.LaneClassificationType.TYPE_INTERSECTION,
            type=l.type,
            original_lane=l
        )
    

@dataclass
class Locator:
    all_lanes: Any # array of all lanes
    external2internal_laneid: dict[Any,int] = field(init=False)
    internal2external_laneid: list[Any] = field(init=False)
    lane_point_distances: list = field(init=False)
    str_tree: shapely.STRtree = field(init=False)
    extended_centerlines: list[shapely.LineString] = field(init=False)
    
    g: nx.DiGraph = field(init=False) # Lane Relation Graph
    xodr2external_laneid: None | dict = None
           
    @classmethod
    def from_mapodr(cls, map):
        all_lanes = [WrappedLane.from_odr_lane(map, l) for l in map.lanes.values()]
        return cls(
            all_lanes = all_lanes,
            xodr2external_laneid = {l.idx: l.idx for l in all_lanes}
        )
        
    def __post_init__(self):
        self.external2internal_laneid = {l.idx: i for i, l in enumerate(self.all_lanes)}
        self.extended_centerlines = [ShapelyTrajectoryTools.extend_linestring(l.centerline) for l in self.all_lanes]
        self.str_tree = shapely.STRtree([l.polygon for l in self.all_lanes])
        self.internal2external_laneid = [l.idx for l in self.all_lanes]
        self.lane_point_distances = [np.unique(shapely.line_locate_point(cl, shapely.points(cl.coords))) for cl in self.extended_centerlines]
        self.g = self._get_routing_graph()
        
    def fullid2xodrid(self, roadlane_id):
        """
        conversion of roadlane id to xodr ids
        """
        if isinstance(roadlane_id[0], list):
            roadlane_id = tuple([tuple(roadlane_id[0]), tuple(roadlane_id[1])])
        else:
            roadlane_id = tuple(roadlane_id)
        l = self.all_lanes[self.external2internal_laneid[roadlane_id]]
        return l.road.odr_idx, l.odr_idx

    def get_route(self, start_id, end_id):
        return nx.shortest_path(self.g, start_id, end_id)
    
    def sts2xys(self, sts):
        xys = np.zeros((len(sts.s), 2), dtype=float)*np.nan
        l_ids = np.array([self.external2internal_laneid[i] for i in sts.roadlane_id.values])
        for l_id in set(l_ids):
            point_idxs = np.argwhere(l_ids==l_id)[:,0]
            rel_sts = sts.isel(dict(time=point_idxs))
            l = self.extended_centerlines[l_id]
            xys[point_idxs, 0], xys[point_idxs, 1] = ShapelyTrajectoryTools.st2xy(l, rel_sts.s.values+ShapelyTrajectoryTools.l_append, rel_sts.t.values)
        return xys

    def xys2sts(self, xys):
        if isinstance(xys, np.ndarray) and xys.ndim==2:
            assert xys.shape[1]==2
            xys = shapely.points(xys)
        lat_distances, lon_distances = self._xys2sts(xys)
        single_lane_association = self.get_single_lane_association(lat_distances)
        sla = np.zeros(len(single_lane_association), dtype=tuple)
        for i, v in enumerate(single_lane_association):
            sla[i]=v
        sts = xr.Dataset(
            {
                's': ('time', [lon_distances[lidx][i] for i, lidx in enumerate(single_lane_association)]),
                't': ('time', [lat_distances[lidx][i] for i, lidx in enumerate(single_lane_association)]),
                'roadlane_id': ('time', sla),
            }
        )
        return sts
    

    def xys2lane_sts(self, lane_id, xys, internal_id=False):
        # xys should be an array of shapely objects or an array of points with dim (n_points, 2)
        # return (n_points, 2) where ret[:,0] is s and ret[:,1] is t 
        if isinstance(xys, np.ndarray) and xys.ndim==2:
            assert xys.shape[1]==2
            xys = shapely.points(xys)
        lid = self.external2internal_laneid[lane_id] if not internal_id else lane_id
        lane_point_distances = self.lane_point_distances[lid]
        sts = ShapelyTrajectoryTools.xy2st(self.extended_centerlines[lid], x_or_xy=xys, line_point_distances=lane_point_distances)
        sts[:,0]-=ShapelyTrajectoryTools.l_append
        return sts
        

    def _xys2sts(self, xys, polys=None):
        # xys should be an array of shapely objects or an array of points with dim (n_points, 2)
        if isinstance(xys, np.ndarray) and xys.ndim==2:
            assert xys.shape[1]==2
            xys = shapely.points(xys)
        else:
            xys = np.array(xys)
        if polys is None:
            polys=xys
        lon_distances = defaultdict(lambda: np.nan * np.ones((len(xys),)))
        lat_distances = defaultdict(lambda: np.nan * np.ones((len(xys),)))
        point_idxs, intersection_lane_ids = self.str_tree.query(polys, predicate='intersects')
        for l_id in set(intersection_lane_ids):
            lps = point_idxs[intersection_lane_ids==l_id]
            lon_distances[self.internal2external_laneid[l_id]][lps] , lat_distances[self.internal2external_laneid[l_id]][lps] = self.xys2lane_sts(l_id, xys[lps], internal_id=True).T
        try:
            no_associations = np.where(np.all(np.isnan(np.stack(list(lon_distances.values()))),axis=0))[0]
        except ValueError:
            # no arrays to stack 
            no_associations = np.array(list(range(len(xys))))
        no_asscociation_idxs, intersection_lane_ids = self.str_tree.query_nearest(polys[no_associations])
        for l_id in set(intersection_lane_ids):
            lps = no_associations[no_asscociation_idxs[intersection_lane_ids==l_id]]
            lon_distances[self.internal2external_laneid[l_id]][lps] , lat_distances[self.internal2external_laneid[l_id]][lps] = self.xys2lane_sts(l_id, xys[lps], internal_id=True).T
  
        no_asscociation_new = np.where(np.all(np.isnan(np.stack(list(lon_distances.values()))),axis=0))[0]

        assert len(no_asscociation_new)==0
        return lat_distances, lon_distances
    
    def _get_routing_graph(self):
        all_lanes = self.all_lanes
        str_tree = self.str_tree
        external2internal_laneid = self.external2internal_laneid
        g = nx.DiGraph()
        for lid, lane in enumerate(all_lanes):
            for external_pid in lane.predecessor_idxs:
                try:
                    g.add_edge(lid, external2internal_laneid[external_pid], label=LaneRelation.predecessor)
                except KeyError:
                    pass
            for external_sid in lane.successor_idxs:
                try:
                    g.add_edge(lid, external2internal_laneid[external_sid], label=LaneRelation.successor)
                except KeyError:
                    pass
            right_neigbours = [i for i in str_tree.query(lane.border_right, predicate='covered_by') if i!=lid]
            left_neigbours = [i for i in str_tree.query(lane.border_left, predicate='covered_by') if i!=lid]
            for rn in right_neigbours:
                g.add_edge(lid, rn)
            for ln in left_neigbours:
                g.add_edge(lid, ln)
        return g
        
    
    def get_single_lane_association(self, traveler_lane_intersections: dict[Any,Any], overlaps: None | dict[Any, float]=None):
        """
        filter traveling path of traveler, so that traveler is not assigned to lanes that are only reachable through a merging or crossing relation
        return format: road, lane
        """
        import networkx as nx
        g = nx.Graph()
        nodes = defaultdict(list)
        for external_lid, v in traveler_lane_intersections.items():
            for timeidx in np.where(~np.isnan(v))[0]:
                nodes[timeidx].append(self.external2internal_laneid[external_lid])
        nodes_per_time = [nodes[i] for i in range(len(nodes))]
        g.add_node('start', pos=(-1, -1))
        g.add_node('end', pos=(len(nodes_per_time), -1))
        for i, nodes_of_time in enumerate(nodes_per_time[:-1]):
            for n in nodes_of_time:
                for next_n in nodes_per_time[i+1]:
                    g.add_node((n,i), pos=(i, n))
                    g.add_node((next_n,i+1), pos=(i+1, next_n))
                    try:
                        if n==next_n:
                            if overlaps is None:
                                weight=1
                            else:
                                weight=1-overlaps[self.internal2external_laneid[n]][i+1]
                        elif any([LaneRelation.neighbour_left in o or LaneRelation.neighbour_right in o for o in self.g.get_edge_data(n, next_n)['label']+self.g.get_edge_data(next_n, n)['label']]):
                            weight=2
                        elif self.g.get_edge_data(n, next_n)['label'] in [LaneRelation.predecessor, LaneRelation.successor] or \
                            self.g.get_edge_data(n, next_n)['label'] in [LaneRelation.predecessor, LaneRelation.successor]:
                            weight=2
                        else:
                            weight=3
                    except Exception:
                        weight=4
                    g.add_edge((n,i), (next_n,i+1), weight=weight)
        for n in nodes_per_time[0]:
            g.add_edge('start', (n,0), weight=1)
        for n in nodes_per_time[-1]:
            g.add_edge((n,len(nodes_per_time)-1), 'end', weight=1)
        sp = nx.shortest_path(g, 'start', 'end', weight='weight')[1:-1]
        fixed_traveler_path = [self.internal2external_laneid[o[0]] for o in sp]
        
        overlaps = [traveler_lane_intersections[lid][i] for i, lid in enumerate(fixed_traveler_path)]
        assert not np.any(np.isnan(overlaps))
        return fixed_traveler_path

    def __repr__(self):
            return f'Locator({len(self.all_lanes)} lanes)<{id(self)}>'
        

