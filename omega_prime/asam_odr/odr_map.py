from dataclasses import dataclass
from pathlib import Path
from typing import Any

import betterosi
import numpy as np
from lxml import etree
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as PltPolygon

from ..map import Map
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
    
    @classmethod
    def from_file(cls, filename, topic='ground_truth_map', is_odr_xml: bool = False, is_mcap: bool = False, step_size=0.1, skip_parse: bool = False):
        if Path(filename).suffix in ['.xodr', '.odr'] or is_odr_xml:
            with open(filename) as f:
                self = cls.create(odr_xml=f.read(), name=Path(filename).stem, step_size=step_size, skip_parse=skip_parse)
            return self
        elif Path(filename).suffix in ['.mcap'] or is_mcap:
            map = next(iter(betterosi.read(filename, mcap_topics=[topic], osi_message_type=betterosi.MapAsamOpenDrive)))
            return cls.create(odr_xml=map.open_drive_xml_content, name=map.map_reference, step_size=step_size, skip_parse=skip_parse)
    
    @classmethod
    def create(cls, odr_xml, name, step_size=.1, skip_parse: bool = False):
        self = cls(
            odr_xml = odr_xml,
            name = name,
            lane_boundaries = {},
            lanes = {},
            step_size=step_size,
            _odr_object = None
        )
        if not skip_parse:
            self.parse()
            return self
        
    def parse(self):
        if self._odr_object is not None:
            return 
        xml = etree.fromstring(self.odr_xml.encode("utf-8"))
        self._odr_object = parse_opendrive(xml)
        self.roads, goerefrence = convert_opendrive(self._odr_object, step_size=self.step_size)
        self.lane_boundaries = {}
        self.lanes = {}
        for rid, r in self.roads.items():
            for bid, b in r.borders.items():
                self.lane_boundaries[(rid, bid)] = b
            for lid, l in r.lanes.items():
                self.lanes[(rid, lid)] = l
 
    def to_osi(self):
        return betterosi.MapAsamOpenDrive(map_reference=self.name, open_drive_xml_content=self.odr_xml)
    
    
    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1,1)
            
        for rid, r in self.roads.items():
            ax.plot(*r.centerline_points[:,1:3].T, c='black')  
        for lid, l in self.lanes.items():
            c = 'blue' if l.type==betterosi.LaneClassificationType.TYPE_UNKNOWN else 'green'
            lb = self.lane_boundaries[l.left_boundary_id]
            rb = self.lane_boundaries[l.right_boundary_id]
            ax.add_patch(PltPolygon(np.concatenate([lb.polyline[:,:2], np.flip(rb.polyline[:,:2], axis=0)]), fc=c, alpha=0.5, ec='black'))
        ax.autoscale()
        ax.set_aspect(1)
        return ax
        

    def setup_lanes_and_boundaries(self):
        self.parse()