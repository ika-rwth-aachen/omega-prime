import logging
from dataclasses import dataclass
from omega_prime.map import Map, Lane, LaneBoundary
from shapely import LineString, Polygon, simplify, make_valid
import numpy as np
from betterosi import LaneClassificationType, LaneClassificationSubtype, LaneBoundaryClassificationType
from pyxodr_omega_prime.road_objects.network import RoadNetwork as PyxodrRoadNetwork
from pyxodr_omega_prime.road_objects.road import Road as PyxodrRoad
from pyxodr_omega_prime.road_objects.lane import Lane as PyxodrLane
from pathlib import Path
from lxml import etree
import pyproj
from .map import ProjectionOffset
from pathlib import Path
import betterosi
from betterosi import MapAsamOpenDrive
from collections import namedtuple
import warnings

logger = logging.getLogger(__name__)


class RoadNetwork(PyxodrRoadNetwork):
    def __init__(
        self,
        xml_string: str,
        resolution: float = 0.1,
        ignored_lane_types: set[str] = set([]),
    ):
        self.root = etree.fromstring(xml_string.encode("utf-8"))
        self.tree = etree.ElementTree(self.root)
        self.resolution = resolution
        self.ignored_lane_types = ignored_lane_types
        self.road_ids_to_object = {}


LANE_BOUNDARY_TYPE_MAP = {
    "none": LaneBoundaryClassificationType.TYPE_NO_LINE,
    "solid": LaneBoundaryClassificationType.TYPE_SOLID_LINE,
    "broken": LaneBoundaryClassificationType.TYPE_DASHED_LINE,
    "botts dots": LaneBoundaryClassificationType.TYPE_BOTTS_DOTS,
    "broken broken": LaneBoundaryClassificationType.TYPE_DASHED_LINE,
    "broken solid": LaneBoundaryClassificationType.TYPE_DASHED_LINE,
    "solid broken": LaneBoundaryClassificationType.TYPE_SOLID_LINE,
    "solid solid": LaneBoundaryClassificationType.TYPE_SOLID_LINE,
    "curb": LaneBoundaryClassificationType.TYPE_CURB,
    "edge": LaneBoundaryClassificationType.TYPE_ROAD_EDGE,
    "grass": LaneBoundaryClassificationType.TYPE_GRASS_EDGE,
    "custom": LaneBoundaryClassificationType.TYPE_OTHER,
}

LANE_SUBTYPE_MAP = {
    "driving": LaneClassificationSubtype.SUBTYPE_NORMAL,
    "shoulder": LaneClassificationSubtype.SUBTYPE_SHOULDER,
    "sidewalk": LaneClassificationSubtype.SUBTYPE_SIDEWALK,
    "walking": LaneClassificationSubtype.SUBTYPE_SIDEWALK,
    "parking": LaneClassificationSubtype.SUBTYPE_PARKING,
    "biking": LaneClassificationSubtype.SUBTYPE_BIKING,
    "stop": LaneClassificationSubtype.SUBTYPE_STOP,
    "restricted": LaneClassificationSubtype.SUBTYPE_RESTRICTED,
    "border": LaneClassificationSubtype.SUBTYPE_BORDER,
    "curb": LaneClassificationSubtype.SUBTYPE_BORDER,
    "exit": LaneClassificationSubtype.SUBTYPE_EXIT,
    "mwyexit": LaneClassificationSubtype.SUBTYPE_EXIT,
    "entry": LaneClassificationSubtype.SUBTYPE_ENTRY,
    "mwyentry": LaneClassificationSubtype.SUBTYPE_ENTRY,
    "onramp": LaneClassificationSubtype.SUBTYPE_ONRAMP,
    "offramp": LaneClassificationSubtype.SUBTYPE_OFFRAMP,
    "connectingramp": LaneClassificationSubtype.SUBTYPE_CONNECTINGRAMP,
}

OTHER_LANE_TYPES = {
    "bus",
    "taxi",
    "hov",
    "median",
    "sliplane",
    "shared",
    "tram",
    "rail",
    "none",
    "roadworks",
    "bidirectional",
    "special1",
    "special2",
    "special3",
}

LANE_TYPE_MAP = {
    betterosi.LaneClassificationType.TYPE_UNKNOWN: ["unknown", "none"],
    betterosi.LaneClassificationType.TYPE_OTHER: [
        "special1",
        "special2",
        "special3",
        "tram",
        "rail",
        "shoulder",
        "median",
    ],
    betterosi.LaneClassificationType.TYPE_DRIVING: [
        "driving",
        "parking",
        "stop",
        "exit",
        "mwyexit",
        "entry",
        "mwyentry",
        "onramp",
        "offramp",
        "connectingramp",
        "bidirectional",
        "hov",
        "taxi",
        "bux",
        "sliplane",
        "shared",
    ],
    betterosi.LaneClassificationType.TYPE_NONDRIVING: [
        "sidewalk",
        "walking",
        "biking",
        "restricted",
        "border",
        "curb",
        "roadworks",
    ],
}

odrlanetype2osilanetype = {odrlt: osilt for osilt, ts in LANE_TYPE_MAP.items() for odrlt in ts}

XodrLaneId = namedtuple("XodrLaneId", ["road_id", "lane_id", "section_id"])
XodrBoundaryId = namedtuple("XodrBoundaryId", ["road_id", "lane_id", "section_id", "side"])


@dataclass(repr=False)
class MapOdr(Map):
    odr_xml: str
    name: str
    step_size: float = 0.01
    _xodr_map: PyxodrRoadNetwork | None = None
    proj_string: str | None = None
    proj_offset: ProjectionOffset | None = None
    projection: pyproj.CRS | None = None
    _supported_file_suffixes = [".xodr", ".mcap", ".odr"]
    _binary_json_identifier = b"xodr"

    @property
    def xodr_map(self):
        if self._xodr_map is None:
            self.parse()
        return self._xodr_map

    @classmethod
    def from_file(
        cls,
        filename,
        topics: list[str] = ["/ground_truth_map", "ground_truth_map"],
        parse_map: bool = False,
        is_odr_xml: bool = False,
        is_mcap: bool = False,
        step_size=0.01,
        ignored_lane_types: set[str] = set([]),
        **kwargs,
    ):
        if Path(filename).suffix in [".xodr", ".odr"] or is_odr_xml:
            with open(filename) as f:
                odr_xml = f.read()
            return cls.create(
                odr_xml=odr_xml,
                name=Path(filename).stem,
                step_size=step_size,
                parse_map=parse_map,
                ignored_lane_types=ignored_lane_types,
            )
        if Path(filename).suffix in [".mcap"] or is_mcap:
            map = next(iter(betterosi.read(filename, mcap_topics=topics, mcap_return_betterosi=False)))
            return cls.create(
                odr_xml=map.open_drive_xml_content, name=map.map_reference, step_size=step_size, parse_map=parse_map
            )

    @property
    def lanes(self):
        if self._lanes is None:
            self.parse()
        return self._lanes

    @lanes.setter
    def lanes(self, val):
        self._lanes = val

    @property
    def lane_boundaries(self):
        if self._lane_boundaries is None:
            self.parse()
        return self._lane_boundaries

    @lane_boundaries.setter
    def lane_boundaries(self, val):
        self._lane_boundaries = val

    @classmethod
    def create(cls, odr_xml, name, step_size=0.01, parse_map: bool = False, ignored_lane_types: set[str] = set([])):
        self = cls(odr_xml=odr_xml, name=name, step_size=step_size, lanes={}, lane_boundaries={})
        self._lane_boundaries = None
        self._lanes = None
        self.ignored_lane_types = ignored_lane_types
        if parse_map:
            self.parse()
        return self

    def parse(self):
        rn = RoadNetwork(self.odr_xml, resolution=self.step_size, ignored_lane_types=self.ignored_lane_types)

        lane_boundaries = {}
        lanes = {}

        # Extract projection information from XML tree
        proj_string = None
        proj_offset = None
        projection = None

        # Get the header element from XML
        header = rn.tree.find("header")
        if header is not None:
            # Get geoReference if it exists
            geo_ref = header.find("geoReference")
            if geo_ref is not None and geo_ref.text:
                proj_string = geo_ref.text.strip()
                try:
                    projection = pyproj.CRS.from_proj4(proj_string)
                except pyproj.exceptions.CRSError as e:
                    logger.warning(f"Failed to parse projection string: {e}")

            # Get offset if it exists
            offset = header.find("offset")
            if offset is not None:
                try:
                    proj_offset = ProjectionOffset(
                        x=float(offset.get("x", "0")),
                        y=float(offset.get("y", "0")),
                        z=float(offset.get("z", "0")),
                        yaw=float(offset.get("hdg", "0")),
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse offset: {e}")

        for road in rn.get_roads():
            lane_idx = 0
            for lane_section_id, lane_section in enumerate(road.lane_sections):
                for lane in lane_section.lanes:
                    boundary_line = getattr(lane, "boundary_line", None)
                    if boundary_line is None or not len(boundary_line):
                        logger.warning(
                            f"Skipping road {road.id} / lane_section {lane_section_id} / lane {lane.id}: missing boundary_line"
                        )
                        continue

                    try:
                        left_boundary = LaneBoundaryXodr.create(
                            lane, road.id, lane.id, lane_section_id, "left", lane_idx=lane_idx
                        )
                        right_boundary = LaneBoundaryXodr.create(
                            lane, road.id, lane.id, lane_section_id, "right", lane_idx=lane_idx
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to create boundaries for road {road.id} / lane_section {lane_section_id} / lane {lane.id}: {e}"
                        )
                        continue

                    lane_boundaries[left_boundary.idx] = left_boundary
                    lane_boundaries[right_boundary.idx] = right_boundary

                    try:
                        lane_obj = LaneXodr.create(lane, road, lane_section_id, lane_idx)
                        lanes[lane_obj.idx] = lane_obj
                    except Exception as e:
                        logger.error(
                            f"Failed to create lane object for road {road.id} / lane_section {lane_section_id} / lane {lane.id}: {e}"
                        )

                    lane_idx += 1

        self._xodr_map = rn
        self.lane_boundaries = lane_boundaries
        self.lanes = lanes
        self.proj_string = proj_string
        self.proj_offset = proj_offset
        self.projection = projection
        for lane in self.lanes.values():
            lane._map = self
            lane._set_boundaries()
            lane._set_polygon()
        for b in self._lane_boundaries.values():
            b._map = self

        return self

    def setup_lanes_and_boundaries(self):
        pass

    def to_file(self, filename: str | Path):
        """Export the current MapOdr to a .xodr file."""
        if isinstance(filename, str):
            filename = Path(filename)

        if filename.is_dir():
            filename = filename / f"{self.name}.xodr"

        if filename.suffix == "":
            filename = filename.with_suffix(".xodr")

        with open(filename, "w") as f:
            f.write(self.odr_xml)

    def to_osi(self):
        return MapAsamOpenDrive(map_reference=self.name, open_drive_xml_content=self.odr_xml)

    def _to_binary_json(self):
        return {b"xodr": self.odr_xml.encode(), b"xodr_name": self.name.encode()}

    @classmethod
    def _from_binary_json(cls, d, parse_map: bool = False, step_size: float = 0.01):
        return cls.create(
            odr_xml=d[b"xodr"].decode(),
            name=d[b"xodr_name"].decode(),
            parse_map=parse_map,
            step_size=step_size,
        )


@dataclass(repr=False)
class LaneBoundaryXodr(LaneBoundary):
    _xodr: PyxodrLane
    idx: XodrBoundaryId

    @classmethod
    def create(
        cls,
        boundary: PyxodrLane,
        road_id: str,
        lane_id: str,
        lane_section_id,
        side: str,
        type: str = None,
        lane_idx: int = None,
    ):
        if side == "left":
            bl = boundary.boundary_line
        elif side == "right":
            bl = boundary.lane_reference_line
        else:
            raise ValueError(f"Invalid side '{side}'. Expected 'left' or 'right'.")

        if len(bl) == 1:
            polyline = LineString([bl[0]] * 2)
        else:
            polyline = LineString(bl)

        if type is None and hasattr(boundary, "lane_xml"):
            type = cls._extract_lane_boundary_type_from_xml(boundary, side)

        lane_boundary_type = cls._determine_lane_boundary_type(type)

        idx = XodrBoundaryId(road_id, lane_id, lane_section_id, side)
        return cls(idx=idx, type=lane_boundary_type, polyline=polyline, _xodr=boundary)

    @staticmethod
    def _extract_lane_boundary_type_from_xml(lane, side: str) -> str:
        if not hasattr(lane, "lane_xml"):
            return None
        road_marks = lane.lane_xml.findall("roadMark")
        try:
            if side == "right" and len(road_marks) > 0:
                if float(road_marks[0].get("sOffset", "inf")) == 0.0:
                    return road_marks[0].get("type")
            elif side == "left" and len(road_marks) > 1:
                return road_marks[1].get("type")
        except Exception:
            return None
        return None

    @staticmethod
    def _determine_lane_boundary_type(boundary_type_str: str) -> LaneBoundaryClassificationType:
        return LANE_BOUNDARY_TYPE_MAP.get(
            (boundary_type_str or "unknown").strip().lower(), LaneBoundaryClassificationType.TYPE_UNKNOWN
        )


@dataclass(repr=False)
class LaneXodr(Lane):
    _xodr: PyxodrLane
    idx: XodrLaneId

    @classmethod
    def create(cls, lane: PyxodrLane, road: PyxodrRoad, lane_section_id: int, lane_idx: int = None):
        idx = XodrLaneId(road.id, lane.id, lane_section_id)
        centre_line = getattr(lane, "centre_line", None)
        if centre_line is None or not len(centre_line):
            raise ValueError(f"Lane {lane.id} has no centre_line")

        centerline = LineString(centre_line[:, :2])
        if not centerline.is_valid:
            centerline = make_valid(centerline)
            warnings.warn(
                f"Needed to make centerline of lane {idx} valid. Most likely, because the OpenDRIVE geometry is translated to a zero length polyline. Try to decrease `step_size`."
            )

        lane_type, lane_subtype = cls._determine_lane_type_and_subtype(lane, road)
        return cls(
            _xodr=lane,
            idx=idx,
            centerline=centerline,
            type=lane_type,
            subtype=lane_subtype,
            successor_ids=[
                XodrLaneId(s.road_id, s.id, s.lane_section_id) for s in set([o[0] for o in lane.successor_data])
            ],
            predecessor_ids=[
                XodrLaneId(p.road_id, p.id, p.lane_section_id) for p in set([o[0] for o in lane.predecessor_data])
            ],
            right_boundary_id=XodrBoundaryId(road.id, lane.id, lane_section_id, side="right"),
            left_boundary_id=XodrBoundaryId(road.id, lane.id, lane_section_id, side="left"),
        )

    @staticmethod
    def _determine_lane_type_and_subtype(lane: PyxodrLane, road: PyxodrRoad):
        is_junction = road.road_xml.get("junction") != "-1"
        lane_type_str = (getattr(lane, "type", "unknown") or "unknown").lower()

        if is_junction:
            lane_type = LaneClassificationType.TYPE_INTERSECTION
        else:
            lane_type = odrlanetype2osilanetype[lane_type_str]

        if lane_type_str in LANE_SUBTYPE_MAP:
            lane_subtype = LANE_SUBTYPE_MAP[lane_type_str]
        elif lane_type_str in OTHER_LANE_TYPES:
            lane_subtype = LaneClassificationSubtype.SUBTYPE_OTHER
        else:
            lane_subtype = LaneClassificationSubtype.SUBTYPE_UNKNOWN

        return lane_type, lane_subtype

    def _set_boundaries(self):
        self.left_boundary = self._map._lane_boundaries[self.left_boundary_id]
        self.right_boundary = self._map._lane_boundaries[self.right_boundary_id]
        return self

    def _set_polygon(self):
        coords = np.concatenate(
            [
                np.asarray(self.left_boundary.polyline.coords),
                np.flip(np.asarray(self.right_boundary.polyline.coords), axis=0),
            ]
        )
        polygon = Polygon(coords)
        if not polygon.is_valid:
            polygon = simplify(polygon, tolerance=0.01)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    raise ValueError(f"Could not compute valid polygon for Lane {self.idx}")
                else:
                    warnings.warn(f"Needed to simplify and buffer polygon for Lane {self.idx}.")
                    pass
            else:
                warnings.warn(f"Needed to simplify polygon for Lane {self.idx}.")
                pass
        self.polygon = polygon
        return self
