from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from matplotlib import pyplot as plt

from omega_prime.mapsegment import MapSegmentation, _save_or_show
from omega_prime.maposicenterlinesegmentation import MapOsiCenterlineSegmentation


class FakeRoadlaneArray:
    def __init__(self, lane_ids):
        self._lane_ids = lane_ids

    def to_numpy(self):
        lane_ids = np.empty(len(self._lane_ids), dtype=object)
        lane_ids[:] = self._lane_ids
        return lane_ids


class FakeSts:
    def __init__(self, lane_ids):
        self._lane_ids = lane_ids

    def __getitem__(self, key):
        assert key == "roadlane_id"
        return FakeRoadlaneArray(self._lane_ids)


class FakeLocator:
    def __init__(self, lane_ids):
        self._lane_ids = lane_ids

    def xys2sts(self, xy):
        assert xy.shape[1] == 2
        return FakeSts(self._lane_ids)


class FakeBaseSegmentation(MapSegmentation):
    def _get_lane_id(self, lane):
        return lane.idx

    def _get_lane_centerline(self, lane):
        return lane.centerline

    def _get_lane_successors(self, lane):
        return []

    def _get_lane_predecessors(self, lane):
        return []

    def _has_traffic_light(self, lane):
        return False

    def _get_traffic_light(self, lane):
        return None

    def _set_lane_on_intersection(self, lane, value):
        lane.on_intersection = value

    def _set_lane_is_approaching(self, lane, value):
        lane.is_approaching = value

    def _get_lane_on_intersection(self, lane):
        return False


def make_segmentation(lanes, lane_ids, lane_segment_dict):
    segmentation = FakeBaseSegmentation.__new__(FakeBaseSegmentation)
    segmentation.lanes = lanes
    segmentation.lane_dict = lanes.copy()
    segmentation.locator = FakeLocator(lane_ids)
    segmentation.lane_segment_dict = lane_segment_dict
    return segmentation


def test_trajectory_segment_detection_splits_on_segment_changes():
    segment_a = SimpleNamespace(idx=0)
    segment_b = SimpleNamespace(idx=1)
    lanes = {
        "lane-a": SimpleNamespace(idx="lane-a"),
        "lane-b": SimpleNamespace(idx="lane-b"),
    }
    segmentation = make_segmentation(
        lanes,
        ["lane-a", "lane-a", "lane-b"],
        {
            "lane-a": SimpleNamespace(segment=segment_a),
            "lane-b": SimpleNamespace(segment=segment_b),
        },
    )
    trajectory = np.array([[0, 0.0, 0.0], [1, 1.0, 0.0], [2, 2.0, 0.0]])

    chunks = segmentation.trajectory_segment_detection(trajectory)

    assert len(chunks) == 2
    assert chunks[0][1] is segment_a
    assert chunks[0][0].tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    assert chunks[1][1] is segment_b
    assert chunks[1][0].tolist() == [[2.0, 2.0, 0.0]]


def test_trajectory_segment_detection_normalizes_osi_lane_ids():
    LaneId = namedtuple("LaneId", ["road_id", "lane_id"])
    located_lane_id = LaneId(5, 42)
    segment = SimpleNamespace(idx=0)
    segmentation = MapOsiCenterlineSegmentation.__new__(MapOsiCenterlineSegmentation)
    lane = SimpleNamespace(idx=located_lane_id)
    segmentation.lanes = {located_lane_id: lane}
    segmentation.lane_dict = {42: lane}
    segmentation.locator = FakeLocator([located_lane_id])
    segmentation.lane_segment_dict = {42: SimpleNamespace(segment=segment)}

    chunks = segmentation.trajectory_segment_detection(np.array([[0, 1.0, 2.0]]))

    assert len(chunks) == 1
    assert chunks[0][1] is segment
    assert chunks[0][0].tolist() == [[0.0, 1.0, 2.0]]


def test_trajectory_segment_detection_uses_full_odr_lane_ids():
    XodrLaneId = namedtuple("XodrLaneId", ["road_id", "lane_id", "section_id"])
    lane_id = XodrLaneId("road-1", "lane-1", 0)
    segment = SimpleNamespace(idx=0)
    segmentation = make_segmentation(
        {lane_id: SimpleNamespace(idx=lane_id)},
        [lane_id],
        {lane_id: SimpleNamespace(segment=segment)},
    )

    chunks = segmentation.trajectory_segment_detection(np.array([[0, 1.0, 2.0]]))

    assert chunks[0][1] is segment


def test_trajectory_segment_detection_raises_for_unknown_lane():
    segmentation = make_segmentation({}, ["missing-lane"], {})

    with pytest.raises(RuntimeError, match="not present in lane_segment_dict"):
        segmentation.trajectory_segment_detection(np.array([[0, 1.0, 2.0]]))


def test_trajectory_segment_detection_raises_for_unassigned_lane():
    lane = SimpleNamespace(idx="lane-a")
    segmentation = make_segmentation(
        {"lane-a": lane},
        ["lane-a"],
        {"lane-a": SimpleNamespace(segment=None)},
    )

    with pytest.raises(RuntimeError, match="has no assigned segment"):
        segmentation.trajectory_segment_detection(np.array([[0, 1.0, 2.0]]))


def test_trajectory_segment_detection_raises_when_lane_object_cannot_be_resolved():
    segment = SimpleNamespace(idx=0)
    segmentation = make_segmentation(
        {},
        ["lane-a"],
        {"lane-a": SimpleNamespace(segment=segment)},
    )

    with pytest.raises(RuntimeError, match="could not be resolved to a lane object"):
        segmentation.trajectory_segment_detection(np.array([[0, 1.0, 2.0]]))


def test_trajectory_segment_detection_raises_for_point_outside_lane_polygon():
    segment = SimpleNamespace(idx=0)
    lane = SimpleNamespace(
        idx="lane-a",
        polygon=shapely.box(0.0, 0.0, 1.0, 1.0),
    )
    segmentation = make_segmentation(
        {"lane-a": lane},
        ["lane-a"],
        {"lane-a": SimpleNamespace(segment=segment)},
    )

    with pytest.raises(RuntimeError, match="outside located lane"):
        segmentation.trajectory_segment_detection(np.array([[0, 2.0, 2.0]]))


def test_trajectory_segment_detection_accepts_lane_polygon_boundary_points():
    segment = SimpleNamespace(idx=0)
    lane = SimpleNamespace(
        idx="lane-a",
        polygon=shapely.box(0.0, 0.0, 1.0, 1.0),
    )
    segmentation = make_segmentation(
        {"lane-a": lane},
        ["lane-a"],
        {"lane-a": SimpleNamespace(segment=segment)},
    )

    chunks = segmentation.trajectory_segment_detection(np.array([[0, 1.0, 1.0]]))

    assert chunks[0][1] is segment


def test_trajectory_segment_detection_validates_input_shape():
    segmentation = make_segmentation({}, [], {})

    with pytest.raises(ValueError, match="shape"):
        segmentation.trajectory_segment_detection([[0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="shape"):
        segmentation.trajectory_segment_detection(np.array([0, 1.0, 2.0]))
    assert segmentation.trajectory_segment_detection(np.empty((0, 3))) == []


def test_save_or_show_creates_missing_suffixless_directory(tmp_path):
    output_dir = tmp_path / "plots"

    plt.figure()
    _save_or_show(output_dir, "example.pdf")

    assert output_dir.is_dir()
    assert (output_dir / "example.pdf").is_file()


def test_save_or_show_accepts_supported_file_paths(tmp_path):
    for suffix in (".pdf", ".png", ".svg"):
        output_file = tmp_path / f"plot{suffix}"

        plt.figure()
        _save_or_show(output_file, "ignored.pdf")

        assert output_file.is_file()


def test_save_or_show_rejects_unsupported_file_suffix(tmp_path):
    plt.figure()

    with pytest.raises(ValueError, match="directory or a file path"):
        _save_or_show(tmp_path / "plot.txt", "ignored.pdf")


def test_plot_intersections_delegates_to_intersections_and_connections():
    calls = []

    class FakeSegment:
        def __init__(self, name):
            self.name = name

        def plot(self, output_plot=None):
            calls.append((self.name, output_plot))

    segmentation = FakeBaseSegmentation.__new__(FakeBaseSegmentation)
    segmentation.intersections = [FakeSegment("intersection")]
    segmentation.isolated_connections = [FakeSegment("connection")]

    segmentation.plot_intersections("target")

    assert calls == [("intersection", "target"), ("connection", "target")]
