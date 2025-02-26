from dataclasses import dataclass
from lxml import etree

from .opendriveparser.parser import parse_opendrive
from .opendriveconverter.converter import convert_opendrive
from typing import Any
import betterosi
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PltPolygon

@dataclass(repr=False)
class MapOdr():
    odr_xml: str
    name: str
    step_size: float = .001
    _parsed: Any = None
    _map: Any = None
    
    @classmethod
    def from_file(cls, filename, topic='ground_truth_map', is_odr_xml: bool = False, is_mcap: bool = False):
        if Path(filename).suffix in ['.xodr', '.odr'] or is_odr_xml:
            with open(filename, "r") as f:
                self = cls(odr_xml=f.read(), name=Path(filename).stem)
            return self
        elif Path(filename).suffix in ['.mcap'] or is_mcap:
            map = next(iter(betterosi.read(filename, mcap_topics=[topic], osi_message_type=betterosi.MapAsamOpenDrive)))
            return cls(odr_xml=map.open_drive_xml_content, name=map.map_reference)
    
    def to_osi(self):
        return betterosi.MapAsamOpenDrive(map_reference=self.name, open_drive_xml_content=self.odr_xml)
    
    def parse(self):
        xml = etree.fromstring(self.odr_xml)
        self._parsed = parse_opendrive(xml)
        return self._parsed

    def get_lanes_and_boundaries(self):
        if self._parsed is None:
            self.parse()
        self._map = convert_opendrive(self._parsed)
        return self._map
    
    def plot(self, ax=None):
        if self._map is None:
            self.get_lanes_and_boundaries()
        if ax is None:
            _, ax = plt.subplots(1,1)
            
        for rid, r in self._map.roads.items():
            ax.plot(*r.centerline_points[:,1:3].T, c='black')
            for lid, l in r.lanes.items():
                c = 'blue' if l.type==betterosi.LaneClassificationType.TYPE_UNKNOWN else 'green'
                lb = self._map.roads[l.left_border_id[0]].borders[l.left_border_id[1]]
                rb = self._map.roads[l.right_border_id[0]].borders[l.right_border_id[1]]
                ax.add_patch(PltPolygon(np.concatenate([lb.polyline[:,:2], np.flip(rb.polyline[:,:2], axis=0)]), fc=c, alpha=0.5, ec='black'))
        ax.autoscale()
        ax.set_aspect(1)
        return ax
        
