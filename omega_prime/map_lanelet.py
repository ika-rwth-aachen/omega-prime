import omega_prime
import betterosi
from warnings import warn
import lanelet2
import lanelet2.core as ltc
from tqdm.auto import tqdm
import shapely
import numpy as np
from dataclasses import dataclass, field


lst = betterosi.LaneClassificationSubtype
lt = betterosi.LaneClassificationType

lanelet2lst = {
    "road": lst.SUBTYPE_NORMAL,
    "highway": lst.SUBTYPE_NORMAL,
    "play_street": lst.SUBTYPE_NORMAL,
    "emergency_lane": lst.SUBTYPE_RESTRICTED,
    "bus_lane": lst.SUBTYPE_RESTRICTED,
    "bicycle_lane": lst.SUBTYPE_BIKING,
    "walkway": lst.SUBTYPE_SIDEWALK,
    "shared_walkway": lst.SUBTYPE_SIDEWALK,
    "parking": lst.SUBTYPE_PARKING,
    "freespace": lst.SUBTYPE_NORMAL,
    "exit": lst.SUBTYPE_EXIT,
    "keepout": lst.SUBTYPE_RESTRICTED,
    "crosswalk": lst.SUBTYPE_NORMAL

}

lst2lanelet = {
    lst.SUBTYPE_UNKNOWN: 'road',
    lst.SUBTYPE_OTHER: 'road',
    lst.SUBTYPE_NORMAL: 'road',
    lst.SUBTYPE_BIKING: 'bicycle_lane',
    lst.SUBTYPE_SIDEWALK: 'walkway',
    lst.SUBTYPE_PARKING: 'parking',
    lst.SUBTYPE_STOP: 'road',
    lst.SUBTYPE_RESTRICTED: 'keepout',
    #lst.SUBTYPE_BORDER: ''
    #lst.SUBTYPE_SHOULDER: ''
    lst.SUBTYPE_EXIT: 'exit',
    lst.SUBTYPE_ENTRY: 'road',
    lst.SUBTYPE_ONRAMP: 'road',
    lst.SUBTYPE_OFFRAMP: 'road',
    lst.SUBTYPE_CONNECTINGRAMP: 'road'
}


lanelet2lt = {
    "road": lt.TYPE_DRIVING,
    "highway": lt.TYPE_DRIVING,
    "play_street": lt.TYPE_DRIVING,
    "emergency_lane": lt.TYPE_NONDRIVING,
    "bus_lane": lt.TYPE_NONDRIVING,
    "bicycle_lane": lt.TYPE_NONDRIVING,
    "walkway": lt.TYPE_NONDRIVING,
    "shared_walkway": lt.TYPE_NONDRIVING,
    "parking": lt.TYPE_DRIVING,
    "freespace": lt.TYPE_DRIVING,
    "exit": lt.TYPE_DRIVING,
    "keepout": lt.TYPE_NONDRIVING,
    "virtual": lt.TYPE_DRIVING,
    "crosswalk": lt.TYPE_DRIVING
}


not_lane_subtypes = [
    "bicycle_parking",
    "pedestrian_seat",
    "vegetation",
    "building",
    "park",
    "intersection",
]


class RoutingGraph:
    def __init__(self, map, location=lanelet2.traffic_rules.Locations.Germany):
        self.map = map
        self.vehicle = lanelet2.routing.RoutingGraph(
            map, lanelet2.traffic_rules.create(location, lanelet2.traffic_rules.Participants.Vehicle)
        )
        self.pedestrian = lanelet2.routing.RoutingGraph(
            map, lanelet2.traffic_rules.create(location, lanelet2.traffic_rules.Participants.Pedestrian)
        )
        self.cycle = lanelet2.routing.RoutingGraph(
            map, lanelet2.traffic_rules.create(location, lanelet2.traffic_rules.Participants.Bicycle)
        )

    def previous(self, l):
        return list(set(self.vehicle.previous(l) + self.pedestrian.previous(l) + self.cycle.previous(l)))

    def following(self, l):
        return list(set(self.vehicle.following(l) + self.pedestrian.following(l) + self.cycle.following(l)))


@dataclass
class MapLanelet(omega_prime.map.Map):
    lanes: dict[int, "LaneLanelet"]
    lane_boundaries: dict[int, "LaneBoundaryLanelet"]
    lanelet_map: lanelet2.core.LaneletMap
    _lanelet_routing: RoutingGraph = field(init=False)

    def setup_lanes_and_boundaries(self):
        pass

    def __post_init__(self):
        self._lanelet_routing = RoutingGraph(self.lanelet_map)

    @classmethod
    def create(cls, path: str) -> "MapLanelet":
        proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
        lanelet_map = lanelet2.io.load(path, proj)

        map = cls({}, {}, lanelet_map=lanelet_map)
        for l in tqdm(lanelet_map.laneletLayer, desc="Lanes"):
            l = LaneLanelet.create(map, l)
            if l is not None:
                map.lanes[l.idx] = l
        for a in tqdm(lanelet_map.areaLayer, desc="Areas"):
            a = LaneLanelet.create(map, a)
            if a is not None:
                map.lanes[a.idx] = a
        return map


class LaneBoundaryLanelet(omega_prime.map.LaneBoundary):
    idx: int
    type: str

    @classmethod
    def create(cls, map: MapLanelet, leftrightBound: lanelet2.core.LineString3d) -> "LaneBoundaryLanelet":
        if leftrightBound.id in map.lane_boundaries:
            return map.lane_boundaries[leftrightBound.id]
        else:
            b = cls(
                idx=leftrightBound.id,
                type=dict(leftrightBound.attributes).get("type", ""),
                polyline=shapely.LineString(np.array([(p.x, p.y, p.z) for p in leftrightBound])),
            )
            map.lane_boundaries[b.idx] = b
            return b


@dataclass(repr=False)
class LaneLanelet(omega_prime.map.Lane):
    idx: int
    type: str
    subtype: str
    polygon: shapely.Polygon
    left_boundary: LaneBoundaryLanelet
    right_boundary: LaneBoundaryLanelet

    @classmethod
    def create(cls, map: MapLanelet, obj: lanelet2.core.Area | lanelet2.core.Lanelet) -> "LaneLanelet":
        rb = None
        lb = None
        polygon = None
        centerline = None
        successor_ids = []
        predecessor_ids = []

        ltype = dict(obj.attributes).get("type", "")
        lsubtype = dict(obj.attributes).get("subtype", "")
        if lsubtype in not_lane_subtypes:
            return None
        type = lanelet2lt.get(lsubtype, None)
        subtype = lanelet2lst.get(lsubtype, None)

        if type is None or subtype is None:
            warn(
                f"Lanelet Lane id={obj.id} could not be mapped to omega_prime with type={ltype} and subtype={lsubtype}."
            )
            return None

        if isinstance(obj, lanelet2.core.Lanelet):
            lb = LaneBoundaryLanelet.create(map, obj.leftBound)
            rb = LaneBoundaryLanelet.create(map, obj.rightBound)
            polygon = shapely.Polygon(np.array([(p.x, p.y, p.z) for p in obj.polygon3d()]))
            centerline = shapely.LineString(np.array([(p.x, p.y, p.z) for p in obj.centerline]))
            successor_ids = (map._lanelet_routing.following(obj),)
            predecessor_ids = map._lanelet_routing.previous(obj)
        elif isinstance(obj, lanelet2.core.Area):
            polygon = shapely.Polygon(np.array([(p.x, p.y, p.z) for p in obj.outerBoundPolygon()]))
            for ip in obj.innerBoundPolygons():
                polygon = polygon.difference(np.array([(p.x, p.y, p.z) for p in ip]))
        return cls(
            idx=obj.id,
            type=type,
            subtype=subtype,
            left_boundary_id=lb.idx if lb else None,
            right_boundary_id=rb.idx if rb else None,
            left_boundary=lb,
            right_boundary=rb,
            polygon=polygon,
            centerline=centerline,
            successor_ids=successor_ids,
            predecessor_ids=predecessor_ids,
        )

def save_lanelet(ll_map, file_dir, projection=None):
    if projection != None:
        raise NotImplementedError
    projection = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
    lanelet2.io.write(file_dir, ll_map, projection)

def map_to_lanelet(map):
    """
    Converts omega prime map to a Lanelet2 map.
    :return: A lanelet2 map
    """

    # Create a new Lanelet2 map
    lanelet_map = ltc.LaneletMap()

    # Create a point layer for storing all map points
    point_layer = {}

    def get_or_create_point(x, y, z=0.0):
        """Returns an existing point or creates a new one"""
        coord = (x, y, z)
        if coord not in point_layer:
            point_layer[coord] = ltc.Point3d(len(point_layer) + 1, *coord)
        return point_layer[coord], coord

    boundary_layer = {}

    def get_or_create_boundary(points):
        points_and_idxs = [get_or_create_point(p[0], p[1]) for p in points]
        idx = tuple([idx for _, idx in points_and_idxs])
        points = tuple([p for p, _ in points_and_idxs])
        if idx not in boundary_layer:
            boundary_layer[idx] = ltc.LineString3d(len(boundary_layer) + 1, points)
        return boundary_layer[idx], idx

    lanelets = []

    laneid2laneletid = {}
    for lane in map.lanes.values():
        
        lb = lane.left_boundary
        rb = lane.right_boundary
        # Create left and right boundaries as Lanelet2 LineStrings
        left_boundary, _ = get_or_create_boundary([(a, b) for a, b in shapely.simplify(lb.polyline, 1).coords])
        right_boundary, _ = get_or_create_boundary([(a, b) for a, b in shapely.simplify(rb.polyline, 1).coords])

        # Create a lanelet using the boundaries
        lanelet_id = len(lanelets) + 1
        laneid2laneletid[lane.idx] = lanelet_id
        lanelet = ltc.Lanelet(
            lanelet_id,
            left_boundary, 
            right_boundary,
            attributes={
                'subtype': lst2lanelet.get(lane.type, 'virtual')
            }
        )
        lanelets.append(lanelet)

        # Add to the map
        lanelet_map.add(lanelet)

    return lanelet_map
    
