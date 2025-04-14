import logging
from dataclasses import dataclass
from omega_prime.map import Map, Lane, LaneBoundary
import shapely
import numpy as np
from matplotlib.patches import Polygon as PltPolygon
import matplotlib.pyplot as plt
from betterosi import LaneClassificationType, LaneClassificationSubtype, LaneBoundaryClassificationType
from pyxodr.road_objects.network import RoadNetwork as PyxodrRoadNetwork
from pyxodr.road_objects.road import Road as PyxodrRoad
from pyxodr.road_objects.lane import Lane as PyxodrLane
from pathlib import Path
from lxml import etree
import pyproj
from .map import ProjectionOffset
from pathlib import Path
import betterosi
from betterosi import MapAsamOpenDrive
from collections import namedtuple

logger = logging.getLogger(__name__)


class RoadNetwork(PyxodrRoadNetwork):
    def __init__(
        self,
        xml_string: str,
        resolution: float = 0.1,
    ):
        self.root = etree.fromstring(xml_string.encode("utf-8"))
        self.tree = etree.ElementTree(self.root)
        self.resolution = resolution
        self.ignored_lane_types = set([])
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


# Define the named tuple
XodrIdx = namedtuple("XodrIdx", ["road_id", "lane_id", "section_idx", "side"])


def make_xodr_idx(road_id, lane_id, section_idx, side=None):
    return XodrIdx(road_id=road_id, lane_id=lane_id, section_idx=section_idx, side=side)


@dataclass(repr=False)
class MapOdr(Map):
    odr_xml: str
    name: str
    step_size: float = 0.01
    _xodr_map: PyxodrRoadNetwork | None = None
    proj_string: str | None = None
    proj_offset: ProjectionOffset | None = None
    projection: pyproj.CRS | None = None

    @property
    def xodr_map(self):
        if self._xodr_map is None:
            self.parse()
        return self._xodr_map

    @classmethod
    def from_file(
        cls,
        filename,
        topic="ground_truth_map",
        parse: bool = False,
        is_odr_xml: bool = False,
        is_mcap: bool = False,
        step_size=0.01,
    ):
        if Path(filename).suffix in [".xodr", ".odr"] or is_odr_xml:
            with open(filename) as f:
                odr_xml = f.read()
            return cls.create(odr_xml=odr_xml, name=Path(filename).stem, step_size=step_size, parse=parse)
        if Path(filename).suffix in [".mcap"] or is_mcap:
            map = next(iter(betterosi.read(filename, mcap_topics=[topic], osi_message_type=betterosi.MapAsamOpenDrive)))
            return cls.create(
                odr_xml=map.open_drive_xml_content, name=map.map_reference, step_size=step_size, parse=parse
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
    def create(cls, odr_xml, name, step_size=0.01, parse: bool = False):
        self = cls(odr_xml=odr_xml, name=name, step_size=step_size, lanes={}, lane_boundaries={})
        self._lane_boundaries = None
        self._lanes = None
        if parse:
            self.parse()
        return self

    def parse(self):
        rn = RoadNetwork(self.odr_xml, resolution=self.step_size)

        lane_boundaries = {}
        lanes = {}

        # Extract projection information from XML tree
        proj_string = None
        proj_offset = None
        projection = None

        # Get the header element from XML
        header = rn.tree.find(".//header")
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
            for lane_section_idx, lane_section in enumerate(road.lane_sections):
                for lane in lane_section.lanes:
                    boundary_line = getattr(lane, "boundary_line", None)
                    if boundary_line is None or not len(boundary_line):
                        logger.warning(
                            f"Skipping road {road.id} / lane_section {lane_section_idx} / lane {lane.id}: missing boundary_line"
                        )
                        continue

                    try:
                        left_boundary = LaneBoundaryXodr.create(
                            lane, road.id, lane.id, lane_section_idx, "left", lane_idx=lane_idx
                        )
                        right_boundary = LaneBoundaryXodr.create(
                            lane, road.id, lane.id, lane_section_idx, "right", lane_idx=lane_idx
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to create boundaries for road {road.id} / lane_section {lane_section_idx} / lane {lane.id}: {e}"
                        )
                        continue

                    lane_boundaries[left_boundary.xodr_idx] = left_boundary
                    lane_boundaries[right_boundary.xodr_idx] = right_boundary

                    try:
                        lane_obj = LaneXodr.create(lane, road, lane_section_idx, lane_idx)
                        lanes[lane_obj.xodr_idx] = lane_obj
                    except Exception as e:
                        logger.error(
                            f"Failed to create lane object for road {road.id} / lane_section {lane_section_idx} / lane {lane.id}: {e}"
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
            lane.set_boundaries()
            lane.set_polygon()
        return self

    def setup_lanes_and_boundaries(self):
        for b in self.lane_boundaries.values():
            b._map = self

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

    def to_hdf(self, filename: Path | str):
        import tables

        # Ensure we have a path to the HDF5 file
        with tables.open_file(filename, mode="a") as h5file:
            # Ensure we have a /map group in the HDF5 file
            try:
                gmap = h5file.get_node("/map")
            except tables.NoSuchNodeError:
                gmap = h5file.create_group("/", "map")

            atom = tables.StringAtom(itemsize=len(self.odr_xml))
            ds = h5file.create_carray(gmap, "odr_xml", atom, shape=(1,))
            ds[0] = self.odr_xml.encode("utf-8")

    def from_hdf(cls, filename: Path | str):
        import tables

        with tables.open_file(filename, mode="r") as h5file:
            odr_xml = h5file.get_node("/map/odr_xml")[0].decode()
        return cls.create(odr_xml, str(filename))


@dataclass(repr=False)
class LaneBoundaryXodr(LaneBoundary):
    _xodr: PyxodrLane
    xodr_idx: str

    @classmethod
    def create(
        cls,
        boundary: PyxodrLane,
        road_id: str,
        lane_id: str,
        lane_section_idx,
        side: str,
        type: str = None,
        lane_idx: int = None,
    ):
        if side == "left":
            polyline = shapely.LineString(boundary.boundary_line)
        elif side == "right":
            polyline = shapely.LineString(boundary.lane_reference_line)
        else:
            raise ValueError(f"Invalid side '{side}'. Expected 'left' or 'right'.")

        if type is None and hasattr(boundary, "lane_xml"):
            type = cls._extract_lane_boundary_type_from_xml(boundary, side)

        lane_boundary_type = cls._determine_lane_boundary_type(type)

        return cls(
            idx=(int(road_id), lane_idx),
            xodr_idx=make_xodr_idx(road_id=road_id, section_idx=lane_section_idx, lane_id=lane_id, side=side),
            type=lane_boundary_type,
            polyline=np.array(polyline.coords),
            _xodr=boundary,
        )

    def plot(self, ax: plt.Axes):
        ax.plot(*np.array(self.polyline[:, :2]).T, color="gray", alpha=0.1)
        ax.set_aspect(1)

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
    xodr_idx: str

    @classmethod
    def create(cls, lane: PyxodrLane, road: PyxodrRoad, lane_section_idx: int, lane_idx: int = None):
        centre_line = getattr(lane, "centre_line", None)
        if centre_line is None or not len(centre_line):
            raise ValueError(f"Lane {lane.id} has no centre_line")

        centerline_2d = np.array([(p[0], p[1]) for p in lane.centre_line])
        lane_type, lane_subtype = cls._determine_lane_type_and_subtype(lane, road)

        return cls(
            _xodr=lane,
            idx=(int(road.id), lane_idx),
            xodr_idx=make_xodr_idx(road_id=road.id, section_idx=lane_section_idx, lane_id=lane.id, side=None),
            centerline=centerline_2d,
            type=lane_type,
            subtype=lane_subtype,
            successor_ids=[(int(road.id), lane_section_idx + 1, successor) for successor in lane.successor_ids],
            predecessor_ids=[(int(road.id), lane_section_idx - 1, predecessor) for predecessor in lane.predecessor_ids],
            right_boundary_id=make_xodr_idx(
                road_id=road.id, section_idx=lane_section_idx, side="right", lane_id=lane.id
            ),
            left_boundary_id=make_xodr_idx(road_id=road.id, section_idx=lane_section_idx, side="left", lane_id=lane.id),
        )

    @staticmethod
    def _determine_lane_type_and_subtype(lane: PyxodrLane, road: PyxodrRoad):
        lane_type = (
            LaneClassificationType.TYPE_UNKNOWN
            if road.road_xml.get("junction") == "-1"
            else LaneClassificationType.TYPE_INTERSECTION
        )
        lane_type_str = (getattr(lane, "type", "unknown") or "unknown").lower()

        if lane_type_str in LANE_SUBTYPE_MAP:
            lane_subtype = LANE_SUBTYPE_MAP[lane_type_str]
        elif lane_type_str in OTHER_LANE_TYPES:
            lane_subtype = LaneClassificationSubtype.SUBTYPE_OTHER
        else:
            lane_subtype = LaneClassificationSubtype.SUBTYPE_UNKNOWN

        return lane_type, lane_subtype

    def set_boundaries(self):
        self.left_boundary = self._map.lane_boundaries[self.left_boundary_id]
        self.right_boundary = self._map.lane_boundaries[self.right_boundary_id]
        return self

    def set_polygon(self):
        coords = np.concatenate(
            [
                self.left_boundary.polyline,
                np.flip(self.right_boundary.polyline, axis=0),
            ]
        )
        polygon = shapely.Polygon(coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        self.polygon = polygon
        return self

    def plot(self, ax: plt.Axes):
        c = "green" if self.type != LaneClassificationType.TYPE_INTERSECTION else "black"
        ax.plot(*np.array(self.centerline).T, color=c, alpha=0.5)
        # try:
        if isinstance(self.polygon, shapely.geometry.MultiPolygon):
            ps = self.polygon.geoms
        else:
            ps = [self.polygon]
        for p in ps:
            ax.add_patch(PltPolygon(p.exterior.coords, fc="blue", alpha=0.2, ec="black"))
