from collections import namedtuple
from types import SimpleNamespace

from omega_prime.maposicenterlinesegmentation import MapOsiCenterlineSegmentation

LaneId = namedtuple("LaneId", ["road_id", "lane_id"])


def make_lane(road_id, lane_id, successor_ids=None, predecessor_ids=None):
    return SimpleNamespace(
        idx=LaneId(road_id, lane_id),
        successor_ids=list(successor_ids or []),
        predecessor_ids=list(predecessor_ids or []),
        on_intersection=False,
        is_approaching=False,
        trafficlight=None,
    )


def make_segmentation(lanes):
    segmentation = MapOsiCenterlineSegmentation.__new__(MapOsiCenterlineSegmentation)
    segmentation.lanes = lanes
    segmentation.lane_dict = {lane.idx.lane_id: lane for lane in lanes.values()}
    segmentation.intersections = []
    segmentation.isolated_connections = []
    segmentation.lane_segment_dict = {}
    segmentation.lane_successors_dict = {}
    segmentation.lane_predecessors_dict = {}
    return segmentation


def test_osi_lane_id_and_topology_accessors_normalize_lane_ids():
    successor = LaneId(10, 2)
    predecessor = LaneId(10, 3)
    lane = make_lane(
        road_id=10,
        lane_id=1,
        successor_ids=[successor, 4],
        predecessor_ids=[predecessor, 5],
    )
    segmentation = make_segmentation({lane.idx: lane})

    assert segmentation._get_lane_id(lane) == 1
    assert segmentation._get_lane_successors(lane) == [2, 4]
    assert segmentation._get_lane_predecessors(lane) == [3, 5]
    assert segmentation._located_lane_id_to_segment_lane_id(LaneId(99, 42)) == 42
    assert segmentation._located_lane_id_to_segment_lane_id(43) == 43


def test_osi_create_lane_segment_dict_uses_bare_lane_ids():
    lane_1 = make_lane(road_id=10, lane_id=1)
    lane_2 = make_lane(road_id=10, lane_id=2)
    segmentation = make_segmentation({lane_1.idx: lane_1, lane_2.idx: lane_2})
    segmentation.lane_dict = {1: lane_1, 2: lane_2}
    intersection = SimpleNamespace(idx=7, lanes=[lane_1])
    connection = SimpleNamespace(idx=8, lanes=[lane_2])
    segmentation.intersections = [intersection]
    segmentation.isolated_connections = [connection]

    segmentation.create_lane_segment_dict()

    assert set(segmentation.lane_segment_dict) == {1, 2}
    assert segmentation.lane_segment_dict[1].segment_idx == 7
    assert segmentation.lane_segment_dict[1].segment is intersection
    assert segmentation.lane_segment_dict[2].segment_idx == 8
    assert segmentation.lane_segment_dict[2].segment is connection


def test_osi_update_road_ids_rekeys_lanes_and_updates_topology_references():
    old_lane_1_id = LaneId(10, 1)
    old_lane_2_id = LaneId(20, 2)
    lane_1 = make_lane(road_id=old_lane_1_id.road_id, lane_id=old_lane_1_id.lane_id, successor_ids=[old_lane_2_id])
    lane_2 = make_lane(road_id=old_lane_2_id.road_id, lane_id=old_lane_2_id.lane_id, predecessor_ids=[old_lane_1_id])
    segmentation = make_segmentation({old_lane_1_id: lane_1, old_lane_2_id: lane_2})
    segment_1 = SimpleNamespace(idx=0)
    segment_2 = SimpleNamespace(idx=1)
    segmentation.lane_segment_dict = {
        1: SimpleNamespace(segment=segment_1),
        2: SimpleNamespace(segment=segment_2),
    }

    segmentation.update_road_ids()

    new_lane_1_id = LaneId(0, 1)
    new_lane_2_id = LaneId(1, 2)
    assert lane_1.idx == new_lane_1_id
    assert lane_2.idx == new_lane_2_id
    assert set(segmentation.lanes) == {new_lane_1_id, new_lane_2_id}
    assert lane_1.successor_ids == [new_lane_2_id]
    assert lane_2.predecessor_ids == [new_lane_1_id]
    assert segmentation.lane_dict == {1: lane_1, 2: lane_2}
    assert segmentation.lane_successors_dict == {1: [2], 2: []}
    assert segmentation.lane_predecessors_dict == {1: [], 2: [1]}


def test_osi_set_lane_intersection_relation_marks_intersection_and_approaching_lanes():
    predecessor_lane = make_lane(road_id=0, lane_id=1)
    intersection_lane = make_lane(road_id=0, lane_id=2, predecessor_ids=[1])
    unrelated_lane = make_lane(road_id=0, lane_id=3)
    segmentation = make_segmentation(
        {
            predecessor_lane.idx: predecessor_lane,
            intersection_lane.idx: intersection_lane,
            unrelated_lane.idx: unrelated_lane,
        }
    )
    segmentation.lane_dict = {
        1: predecessor_lane,
        2: intersection_lane,
        3: unrelated_lane,
    }
    segmentation.intersections = [SimpleNamespace(lanes=[intersection_lane])]

    segmentation.set_lane_intersection_relation()

    assert intersection_lane.on_intersection is True
    assert intersection_lane.is_approaching is False
    assert predecessor_lane.on_intersection is False
    assert predecessor_lane.is_approaching is True
    assert unrelated_lane.on_intersection is False
    assert unrelated_lane.is_approaching is False
