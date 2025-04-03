from dataclasses import dataclass
from pathlib import Path
from typing import Any

import betterosi
import numpy as np
import pyproj
import shapely
from lxml import etree
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as PltPolygon

from ..map import Map, ProjectionOffset
from .opendriveconverter.converter import convert_opendrive
from .opendriveparser.elements.openDrive import OpenDrive
from .opendriveparser.parser import parse_opendrive


@dataclass(repr=False)
class MapOdr(Map):
    odr_xml: str
    name: str
    roads: dict[Any, Any] | None = None
    _odr_object: OpenDrive | None = None
    step_size: float = 0.1
    proj_string: str | None = None
    proj_offset: ProjectionOffset | None = None
    projection: pyproj.CRS | None = None

    @classmethod
    def from_file(
        cls,
        filename,
        topic="ground_truth_map",
        is_odr_xml: bool = False,
        is_mcap: bool = False,
        step_size=0.1,
        skip_parse: bool = False,
    ):
        if Path(filename).suffix in [".xodr", ".odr"] or is_odr_xml:
            with open(filename) as f:
                self = cls.create(
                    odr_xml=f.read(), name=Path(filename).stem, step_size=step_size, skip_parse=skip_parse
                )
            return self
        elif Path(filename).suffix in [".mcap"] or is_mcap:
            map = next(iter(betterosi.read(filename, mcap_topics=[topic], osi_message_type=betterosi.MapAsamOpenDrive)))
            return cls.create(
                odr_xml=map.open_drive_xml_content, name=map.map_reference, step_size=step_size, skip_parse=skip_parse
            )

    @classmethod
    def create(cls, odr_xml, name, step_size=0.1, skip_parse: bool = False):
        self = cls(odr_xml=odr_xml, name=name, lane_boundaries={}, lanes={}, step_size=step_size, _odr_object=None)
        if not skip_parse:
            self.parse()
            return self

    def parse(self):
        if self._odr_object is not None:
            return
        xml = etree.fromstring(self.odr_xml.encode("utf-8"))
        self._odr_object = parse_opendrive(xml)
        self.proj_string = self._odr_object.header.geo_reference
        if self.proj_string is not None:
            try:
                self.projection = pyproj.CRS.from_proj4(self.proj_string)
            except pyproj.exceptions.CRSError as e:
                raise pyproj.exceptions.CRSError(
                    "Povided ASAM OpenDRIVE XML does not contain a valid georeference!"
                ) from e
        self.proj_offsets = self._odr_object.header.offset
        self.roads = convert_opendrive(self._odr_object, step_size=self.step_size)
        self.lane_boundaries = {}
        self.lanes = {}
        for rid, r in self.roads.items():
            for bid, b in r.borders.items():
                self.lane_boundaries[(rid, bid)] = b
            for lid, l in r.lanes.items():
                self.lanes[(rid, lid)] = l

    def to_file(self, filename: Path | None = None):
        if filename is None:
            p = Path(self.name)
        else:
            p = Path(filename)
            if p.is_dir():
                p = p / f"{self.name}"
        if p.suffix == "":
            p = Path(str(p) + ".xodr")
        with open(p, "w") as f:
            f.write(self.odr_xml)
        print(f"Extracted {p}")

    def to_osi(self):
        return betterosi.MapAsamOpenDrive(map_reference=self.name, open_drive_xml_content=self.odr_xml)

    def to_hdf(self, filename):
        import tables

        with tables.open_file(filename, mode="a") as h5file:
            try:
                gmap = h5file.get_node("/map")
            except tables.NoSuchNodeError:
                gmap = h5file.create_group("/", "map")
            atom = tables.StringAtom(itemsize=len(self.odr_xml))
            ds = h5file.create_carray(gmap, "odr_xml", atom, shape=(1,))
            ds[0] = self.odr_xml.encode("utf-8")

    @classmethod
    def from_hdf(cls, filename):
        import tables

        with tables.open_file(filename, mode="r") as h5file:
            odr_xml = h5file.get_node("/map").odr_xml[0].decode()
        return cls.create(odr_xml, str(filename))

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)

        for rid, r in self.roads.items():
            ax.plot(*r.centerline_points[:, 1:3].T, c="black")
        for lid, l in self.lanes.items():
            c = "blue" if l.type == betterosi.LaneClassificationType.TYPE_UNKNOWN else "green"
            lb = self.lane_boundaries[l.left_boundary_id]
            rb = self.lane_boundaries[l.right_boundary_id]
            ax.add_patch(
                PltPolygon(
                    np.concatenate([lb.polyline[:, :2], np.flip(rb.polyline[:, :2], axis=0)]),
                    fc=c,
                    alpha=0.5,
                    ec="black",
                )
            )
        ax.autoscale()
        ax.set_aspect(1)
        return ax

    def setup_lanes_and_boundaries(self):
        self.parse()
        for lid, l in self.lanes.items():
            l.idx = lid
            l.left_boundary = self.lane_boundaries[l.left_boundary_id]
            l.right_boundary = self.lane_boundaries[l.right_boundary_id]

            l.centerline = (
                l.left_boundary.polyline
                + (np.diff(np.stack([l.left_boundary.polyline, l.right_boundary.polyline]), axis=0) / 2)[0, ...]
            )
            # l.centerline = shapely.LineString(l.centerline)

            poly_points = np.concatenate(
                [l.left_boundary.polyline[:, :2], np.flip(l.right_boundary.polyline[:, :2], axis=0)]
            )
            l.polygon = shapely.Polygon(poly_points)
            if not l.polygon.is_valid:
                l.polygon = shapely.convex_hull(l.polygon)
            l.start_points = np.stack([l.left_boundary.polyline[0, :2], l.right_boundary.polyline[0, :2]])
            l.end_points = np.stack([l.left_boundary.polyline[-1, :2], l.right_boundary.polyline[-1, :2]])
