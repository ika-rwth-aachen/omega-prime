from dataclasses import dataclass
from typing import Any

from ..logger import logger
from .elements.junction import setup_connections
from .elements.road import process_road


@dataclass(repr=False)
class Map:
    roads: dict
    georeference: Any = None


def convert_opendrive(my_opendrive, step_size=0.1) -> tuple[dict, tuple[float,float,str]]:
    """
    Converts the opendrive XML input into a OMEGAFormat road network
    :param my_opendrive: opendrive XML
    :param step_size: resolution in which the maps points are sampled in meters
    :return: returns the road network
    """

    # create roads object and set geo reference
    my_roads = {}
    #georeference = get_georeference(my_opendrive.header)

    # tables needed for matching their ids in open drive to VVM format since many connections can only be set later
    lookup_table = []   # [road id | lane_section | laneID | road id VVM | lane id VVM]
    reference_table = []    # [omega signal id | omega road id | odr signal id | odr reference id | odr connected_to id]

    logger.info(f'Starting conversion of {len(my_opendrive.roads)} opendrive roads into OMEGAFormat.')

    # loop over individual opendrive roads and add them to omega roads dictionary in my_roads
    for road in my_opendrive.roads:
        my_roads, lookup_table, reference_table = process_road(my_roads, road, my_opendrive, step_size, lookup_table, reference_table)

    detected_roads = list(my_roads.keys())
    my_roads = {rid: r for rid, r in my_roads.items() if len(r.lanes)>0}
    no_lane_roads = [rid for rid in detected_roads if rid not in my_roads.keys()]
    if len(no_lane_roads)>0:
        print(f"There are roads with no lanes {no_lane_roads}")
    # setting up road network information via connections and references
    logger.info('Setting up roads connections.')
    my_roads = setup_connections(my_roads, lookup_table, my_opendrive)

    return my_roads
