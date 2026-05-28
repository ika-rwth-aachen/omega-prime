from collections import namedtuple
from types import SimpleNamespace

import pytest

from omega_prime import mapodrsegmentation
from omega_prime.mapodrsegmentation import MapODRSegmentation


def test_identify_segments_uses_unique_indices_when_odr_ids_overlap(monkeypatch):
    class FakeIntersection:
        def __init__(self, lanes, idx=None, concave_hull_ratio=0.3, odr_junction_id=None):
            self.lanes = lanes
            self.idx = idx
            self.odr_junction_id = odr_junction_id

    class FakeConnection:
        def __init__(self, lanes, idx=None, concave_hull_ratio=0.3, odr_road_id=None):
            self.lanes = lanes
            self.idx = idx
            self.odr_road_id = odr_road_id

    monkeypatch.setattr(mapodrsegmentation, "IntersectionOdr", FakeIntersection)
    monkeypatch.setattr(mapodrsegmentation, "ConnectionSegmentOdr", FakeConnection)

    LaneId = namedtuple("LaneId", ["road_id", "lane_id", "section_id"])
    junction_lane = SimpleNamespace(
        idx=LaneId("7", "1", "0"),
        successor_ids=[],
        predecessor_ids=[],
    )
    road_lane = SimpleNamespace(
        idx=LaneId("1", "1", "0"),
        successor_ids=[],
        predecessor_ids=[],
    )

    segmentation = MapODRSegmentation.__new__(MapODRSegmentation)
    segmentation.map = SimpleNamespace(
        xodr_map=SimpleNamespace(
            get_roads=lambda: [
                SimpleNamespace(id="7", road_xml={"junction": "1"}),
                SimpleNamespace(id="1", road_xml={"junction": "-1"}),
            ]
        )
    )
    segmentation.lanes = {junction_lane.idx: junction_lane, road_lane.idx: road_lane}
    segmentation.concave_hull_ratio = 0.3
    segmentation.intersections = []
    segmentation.isolated_connections = []
    segmentation.lane_dict = {}
    segmentation.lane_successors_dict = {}
    segmentation.lane_predecessors_dict = {}
    segmentation.lane_segment_dict = {}
    segmentation.segment_by_road_id = {}
    segmentation.segments = []

    segmentation.identify_segments()

    assert [segment.idx for segment in segmentation.segments] == [0, 1]
    assert segmentation.intersections[0].odr_junction_id == "1"
    assert segmentation.isolated_connections[0].odr_road_id == "1"
    assert segmentation.get_segment("7") is segmentation.intersections[0]
    assert segmentation.get_segment("1") is segmentation.isolated_connections[0]


def test_identify_segments_raises_when_lanes_remain_unassigned(monkeypatch):
    class FailingIntersection:
        def __init__(self, lanes, idx=None, concave_hull_ratio=0.3, odr_junction_id=None):
            raise ValueError("bad geometry")

    monkeypatch.setattr(mapodrsegmentation, "IntersectionOdr", FailingIntersection)

    LaneId = namedtuple("LaneId", ["road_id", "lane_id", "section_id"])
    junction_lane = SimpleNamespace(
        idx=LaneId("7", "1", "0"),
        successor_ids=[],
        predecessor_ids=[],
    )

    segmentation = MapODRSegmentation.__new__(MapODRSegmentation)
    segmentation.map = SimpleNamespace(
        xodr_map=SimpleNamespace(get_roads=lambda: [SimpleNamespace(id="7", road_xml={"junction": "1"})])
    )
    segmentation.lanes = {junction_lane.idx: junction_lane}
    segmentation.concave_hull_ratio = 0.3
    segmentation.intersections = []
    segmentation.isolated_connections = []
    segmentation.lane_dict = {}
    segmentation.lane_successors_dict = {}
    segmentation.lane_predecessors_dict = {}
    segmentation.lane_segment_dict = {}
    segmentation.segment_by_road_id = {}
    segmentation.segments = []

    with pytest.raises(RuntimeError, match="did not assign all lanes") as exc_info:
        segmentation.identify_segments()

    message = str(exc_info.value)
    assert "ODR junction 1" in message
    assert "unassigned lanes" in message


def test_odr_plot_lane_label_uses_road_and_lane_id():
    LaneId = namedtuple("LaneId", ["road_id", "lane_id", "section_id"])
    lane = SimpleNamespace(idx=LaneId("road-7", "lane-2", "0"))
    segmentation = MapODRSegmentation.__new__(MapODRSegmentation)

    assert segmentation._plot_lane_label(lane) == "road-7/lane-2"
