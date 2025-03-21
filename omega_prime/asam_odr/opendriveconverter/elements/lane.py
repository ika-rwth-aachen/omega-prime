import copy
from dataclasses import dataclass
from typing import Any

import betterosi
import numpy as np

from .... import map
from ...logger import logger
from .centerLinePoints import calculate_s_index


@dataclass(repr=False)
class LaneOdr(map.Lane):
    type: Any
    subtype: Any
    
    
@dataclass(repr=False)
class LaneBoundaryOdr(map.LaneBoundary):
    polyline: Any
    
def lane_type_to_boundary_mapping(value: str):
    boundary_lane_types = {
        "curb": betterosi.LaneBoundaryClassificationType.TYPE_UNKNOWN,
        "border": betterosi.LaneBoundaryClassificationType.TYPE_UNKNOWN
    }
    if value.lower() in boundary_lane_types.keys():
        return boundary_lane_types.get(value.lower())
    
def check_lane_type(lane_type):
    """
    Checks the lane type for such types that are not a lane in omega
    :param lane_type:
    :return:
    """
    not_a_lane = False
    is_a_boundary = False
    new_type = betterosi.LaneClassificationType.TYPE_UNKNOWN
    if lane_type_to_boundary_mapping(lane_type) is not None:
        new_type = lane_type_to_boundary_mapping(lane_type)
        not_a_lane = True
        is_a_boundary = True
    return new_type, not_a_lane, is_a_boundary   
 
def insert_centerline_as_border(start_point_index, end_point_index, center_line_points, my_road):
    new_idx = len(my_road.borders)
    my_road.borders[new_idx] = LaneBoundaryOdr(
        polyline=np.stack([
                center_line_points[start_point_index:end_point_index + 1, 1],
                center_line_points[start_point_index:end_point_index + 1, 2],
                center_line_points[start_point_index:end_point_index + 1, 4]]).T,
        idx = new_idx,
        type=None
    )
    return my_road


def calculate_borders(lane_section, center_line_points, end_point_index, start_point_index, my_road, xodr_road, step_size):
    """
    -   this assumes the traditional way in openDRIVE using center line and width (borders are also possible in newer
        versions, but not implemented here)
    -   go over all lanes within lane section (start with most inner on for left and right --> center)
    -   reuse previous border
    -   first border is center (always needs to be present)
    -   center lane should have no width --> should already be calculated in center_line_points
    :param lane_section: lane_section which borders have to be calculated (section along s-axis)
    :param center_line_points: lanes centerline
    :param end_point_index: last points index of lane section in center_line_points table
    :param start_point_index: first points index of lane section in center_line_points table
    :param my_road: road object
    :return: road object now including lanes with width
    """
    my_road = insert_centerline_as_border(start_point_index, end_point_index, center_line_points, my_road)
    # do same for left and right lanes
    sides = dict(left=lane_section.left_lanes, right=lane_section.right_lanes)

    for sideTag, lane_side in sides.items():
        # save center as previous border
        # copy.copy makes an actual copy of the data instead of referencing the original data as done with equals in
        # order to keep the original data available throughout the whole function
        pos_previous = copy.copy(center_line_points[start_point_index:end_point_index+1, [1, 2, 4]]) # x,y,heading

        # get lane sides
        lane_order = np.empty([len(lane_side), 2])
        for count, lane in enumerate(lane_side):
            # get correct order according to ids
            lane_order[count] = [abs(int(lane.id)), count]
        # sort - order needs to be from inner lane to outer lane to reuse previous border
        sorted_lane_order = lane_order[np.argsort(lane_order[:, 0])]

        # find lane_section.s starting index to adapt indexing to pos-Arrays which are cut to sections shape
        lane_section_s_index = calculate_s_index(center_line_points, lane_section.s)

        # run over lanes of one side in correct order
        for row in sorted_lane_order:
            current_lane = lane_side[int(row[1])]
            pos = np.zeros(shape=(end_point_index - start_point_index + 1, 3), dtype=float)
            # starting point of width element - for first entry it is starting point, later on it is the end-point
            # of the previous width entry
            
            subsection_start_index = copy.copy(start_point_index)
            if len(current_lane.width) == 0: 
                raise ValueError(f'No lane width element in lane with id {lane.id}. A default value is set but should be corrected in the openDrive file!') 
            
            for i in range(0, len(current_lane.width)):
                if i < len(current_lane.width) - 1:
                    # if not last entry, find s_index for endpoint of section in center_line_points
                    subsection_end_index = calculate_s_index(center_line_points,
                                                            lane_section.s + current_lane.width[i + 1].s_offset)
                else:
                    # for the last entry it is the end_point_index
                    subsection_end_index = end_point_index

                # lane width is applied between width elements start_index and end_index which are calculated before
                pos = add_width(
                    center_line_points, 
                    current_lane.width[i],
                    subsection_start_index,
                    subsection_end_index, 
                    lane_section_s_index, 
                    pos_previous, 
                    pos, 
                    sideTag
                )
                subsection_start_index = subsection_end_index + 1

            new_id = len(my_road.borders)
            my_border = LaneBoundaryOdr(
                idx=new_id, 
                polyline=pos,
                type=None)
            my_road.borders[new_id] = my_border

            # set previous for next lane as starting point
            pos_previous = copy.copy(pos)

    return my_road


def add_width(center_line_points, current_lane_width, index_from, index_to, index_section_start, pos_previous, pos, orientation):
    """
    :param center_line_points:
    :param current_lane_width: lane_width element
    :param index_from: s-index to start from
    :param index_to: s-index to end with
    :param index_section_start: s-index of the lane_sections starting point
    :param pos_previous: previous borders x,y,z coordinates
    :param pos: x,y,z coordinates to be calculated
    :param orientation: left or right
    :return: coordinates of the border
    """
    j = np.arange(index_from, index_to+1)
    delta_s = center_line_points[j, 0] - current_lane_width.s_offset - center_line_points[index_section_start, 0]
    current_lane_width_value = current_lane_width.a + current_lane_width.b * delta_s + current_lane_width.c * np.pow(delta_s, 2) + current_lane_width.d * np.pow(delta_s, 3)

    if orientation == 'left':
        angle_offset = np.pi/2
    else:
        # right
        angle_offset = -np.pi/2
    pos[j-index_section_start, 0] = \
        pos_previous[j-index_section_start, 0] + \
        + current_lane_width_value * np.cos(
            center_line_points[j-index_section_start, 3] + angle_offset)
    pos[j-index_section_start, 1] = \
        pos_previous[j-index_section_start, 1] \
        + current_lane_width_value * np.sin(
            center_line_points[j-index_section_start, 3] + angle_offset)
    pos[j-index_section_start, 2] = \
        pos_previous[j-index_section_start, 2] \
        +  current_lane_width_value * np.sin(
            center_line_points[j-index_section_start, 5])
    return pos


def calculate_lanes(lane_section, my_road, opendrive_road_id, opendrive_lanesection_id, lookup_table, vvm_road_id,
                    lane_class, lane_subtype):
    """
    get actual lanes, pay attention to correct direction
    :param lane_section:
    :param my_road:
    :param opendrive_road_id:
    :param opendrive_lanesection_id:
    :param lookup_table: [opendrive_road_id, opendrive_lanesection_id, opendrive_lane_id, vvm_road_id, vvm_lane_id]
    :param vvm_road_id:
    :param lane_class:
    :param lane_subtype:
    :return:
    """
    # get number of left and right lanes from lane_section
    number_left_lanes = len(lane_section.left_lanes)
    number_right_lanes = len(lane_section.right_lanes)

    # sanity check: number left lanes+ number right lanes should be number of borders -1
    if len(my_road.borders) - 1 != number_right_lanes + number_left_lanes:
        logger.error('Number of borders and number of right and left lane does not match up')

    # left lanes
    # same order as borders need to be used (from inner to outer border)
    lane_order = np.empty([len(lane_section.left_lanes), 2])
    for count, lane in enumerate(lane_section.left_lanes):
        # get correct order according to ids
        lane_order[count] = [abs(int(lane.id)), count]
    # sort
    sorted_lane_order = lane_order[np.argsort(lane_order[:, 0])]

    count = 0
    # set all left lanes with correct borders
    for row in sorted_lane_order:
        # checks if the lane type implies a conversion into another element, the lane still has to be set because
        # otherwise referencing of borders leads to geometrical failures when setting up road network
        # lanes will be set to type freespace and objects or boundaries are placed on top of the freespace area
        new_type, not_a_lane, is_a_boundary = check_lane_type(lane_section.left_lanes[int(row[1])].type)
        if not_a_lane:
            if is_a_boundary:
                pass    # conversion is done in calculate_boundaries
            else:
                raise RuntimeError()
            new_type = betterosi.LaneClassificationType.TYPE_UNKNOWN # ReferenceTypes.LaneType.FREESPACE

        my_road, lookup_table = set_lanes(my_road, False, count, count + 1, lane_section.left_lanes[int(row[1])].id,
                                          opendrive_road_id, opendrive_lanesection_id, lookup_table, vvm_road_id,
                                          new_type, lane_class, lane_subtype)

        count += 1

    # right lanes
    # same order as borders need to be used (from inner to outer border)
    lane_order = np.empty([len(lane_section.right_lanes), 2])
    for count, lane in enumerate(lane_section.right_lanes):
        # get correct order according to ids
        lane_order[count] = [abs(int(lane.id)), count]
    # sort
    sorted_lane_order = lane_order[np.argsort(lane_order[:, 0])]

    count = 0
    for row in sorted_lane_order:
        # see above comment for left lanes
        new_type, not_a_lane, is_a_boundary = check_lane_type(lane_section.right_lanes[int(row[1])].type)
        if not_a_lane:
            if is_a_boundary:
                pass    # conversion is done in calculate_boundaries
            else:
                raise RuntimeError()
            new_type = betterosi.LaneClassificationType.TYPE_UNKNOWN #ReferenceTypes.LaneType.FREESPACE

        # first right lane (needs to start with center line again)
        if count == 0:
            # set first right lane with correct borders
            my_road, lookup_table = set_lanes(my_road, True, 0, number_left_lanes + 1,
                                              lane_section.right_lanes[int(row[1])].id, opendrive_road_id,
                                              opendrive_lanesection_id, lookup_table, vvm_road_id, new_type,
                                              lane_class, lane_subtype)
            count += 1
        else:
            # set other right lanes with correct borders
            my_road, lookup_table = set_lanes(my_road, True, count+number_left_lanes, count+number_left_lanes+1,
                                              lane_section.right_lanes[int(row[1])].id, opendrive_road_id,
                                              opendrive_lanesection_id, lookup_table, vvm_road_id, new_type,
                                              lane_class, lane_subtype)
            count += 1

    return my_road, lookup_table


def set_lanes(my_road, direction_correct, left_index, right_index, opendrive_lane_id, opendrive_road_id, opendrive_lanesection_id, lookup_table, vvm_road_id, lane_type, lane_class, lane_subtype):

    my_lane = LaneOdr(
        idx = None,
        centerline=None,
        type=lane_type,
        subtype=lane_subtype,
        left_boundary_id = (vvm_road_id, left_index),
        right_boundary_id = (vvm_road_id, right_index),
        predecessor_ids=set(),
        successor_ids=set()
    )
    # set lane class (none, intersection or roundabout)
    if lane_class["intersection"] or lane_class["roundabout"]:
        my_lane.classification = betterosi.LaneClassificationType.TYPE_INTERSECTION

    # add lane to road
    my_road.lanes.update({len(my_road.lanes): my_lane})

    # update lookup table
    new_lane_id = (vvm_road_id, len(my_road.lanes)-1)
    lookup_table.append([opendrive_road_id, opendrive_lanesection_id, opendrive_lane_id, *new_lane_id])

    return my_road, lookup_table


def get_lane_subtype(road):
    """
    Checks if the road is on a bridge or tunnel. The corresponding lane subtype or instead the default value is set.
    :param lane_subtype:
    :param road:
    :return:
    """
    lane_subtype = None
    if road.objects is not None:

        if road.objects.bridge is not None:
            lane_subtype = betterosi.LaneClassificationSubtype.SUBTYPE_UNKNOWN

        elif road.objects.tunnel is not None:
            lane_subtype = betterosi.LaneClassificationSubtype.SUBTYPE_UNKNOWN

    else:
        # this is the default 0 value for subtype
        lane_subtype = betterosi.LaneClassificationSubtype.SUBTYPE_UNKNOWN
    return lane_subtype


def get_lane_class(junction, junction_group):
    """
    get lane class (junction etc.)
    :param junction: default string for road.junction is '-1' (if it is no intersection), else it is the junction id
    :param junction_group: two or more junctions grouped indicate a roundabout
    :return:
    """
    lane_class = {
        "intersection": False,
        "roundabout": False,
    }
    if junction != -1:
        lane_class["intersection"] = True
        if find_roundabout(junction_group) is not None:
            roundabout_list = find_roundabout(junction_group)
            if junction in roundabout_list:
                lane_class["roundabout"] = True

    return lane_class


def find_roundabout(junction_groups):
    """
    find roundabouts in the my_opendrive.junction_group, returns a list of all junctions id's being part of a roundabout
    :param junction_groups:
    :return:
    """
    if junction_groups is not None:
        roundabout_list = []
        for junction_group in junction_groups:
            if junction_group.type == "roundabout":
                # can be roundabout or unknown which is covered as a normal junction
                for junction_reference in junction_group:
                    roundabout_list.append(junction_reference.reference)  # reference is the junction id
            return roundabout_list
