from dataclasses import dataclass
from lxml import etree

from .opendriveparser.parser import parse_opendrive
from .opendriveconverter.converter import convert_opendrive
from .opendriveparser.elements.openDrive import OpenDrive
from ..map import Map
from typing import Any
import betterosi
from pathlib import Path
from matplotlib import pyplot as plt

@dataclass(repr=False)
class MapOdr(Map):
    odr_xml: str
    name: str
    roads: dict[Any,Any]
    _odr_objects: OpenDrive
    step_size: float = .01
    
    @classmethod
    def from_file(cls, filename, topic='ground_truth_map', is_odr_xml: bool = False, is_mcap: bool = False, step_size=0.001):
        if Path(filename).suffix in ['.xodr', '.odr'] or is_odr_xml:
            with open(filename, "r") as f:
                self = cls.create(odr_xml=f.read(), name=Path(filename).stem, step_size=step_size)
            return self
        elif Path(filename).suffix in ['.mcap'] or is_mcap:
            map = next(iter(betterosi.read(filename, mcap_topics=[topic], osi_message_type=betterosi.MapAsamOpenDrive)))
            return cls.create(odr_xml=map.open_drive_xml_content, name=map.map_reference, step_size=step_size)
    
    @classmethod
    def create(cls, odr_xml, name, step_size=.01):
        xml = etree.fromstring(odr_xml)
        odr_objects = parse_opendrive(xml)
        
        roads, goerefrence = convert_opendrive(odr_objects, step_size=step_size)
        
        lane_boundaries = {}
        lanes = {}
        for rid, r in roads.items():
            for bid, b in r.borders.items():
                lane_boundaries[(rid, bid)] = b
            for lid, l in r.lanes.items():
                lanes[(rid, lid)] = l
        return cls(
            roads = roads,
            odr_xml = odr_xml,
            name = name,
            lane_boundaries = lane_boundaries,
            lanes = lanes,
            _odr_objects = odr_objects
        )
 
    def to_osi(self):
        return betterosi.MapAsamOpenDrive(map_reference=self.name, open_drive_xml_content=self.odr_xml)
    
    
    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1,1)
            
        for rid, r in self.roads.items():
            ax.plot(*r.centerline_points[:,1:3].T, c='black')  
            for lid, l in r.lanes.items():
                pass
                #c = 'blue' if l.type==betterosi.LaneClassificationType.TYPE_UNKNOWN else 'green'
                #lb = self.roads[l.left_boundary_id[0]].borders[l.left_boundary_id[1]]
                #rb = self.roads[l.right_boundary_id[0]].borders[l.right_boundary_id[1]]
                #ax.add_patch(PltPolygon(np.concatenate([lb.polyline[:,:2], np.flip(rb.polyline[:,:2], axis=0)]), fc=c, alpha=0.5, ec='black'))

        ax.autoscale()
        ax.set_aspect(1)
        return ax
        

    def setup_lanes_and_boundaries(self):
        pass