import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from lxml import etree
from shapely import LineString

from omega_prime.map_odr import MapOdr

FILES = Path(__file__).parent / "files"


def test_lane_section_shorter_than_step_size_is_dropped_and_bridged():
    # pyxodr gives a lane section without samples zero coordinates: a lane at the origin without geometry
    tree = etree.parse(str(FILES / "straight_500m.xodr"))
    lanes = tree.find(".//road/lanes")
    for s in (100.0, 100.5):
        section = copy.deepcopy(lanes.find("laneSection"))
        section.set("s", str(s))
        lanes.append(section)
    for lane in lanes.iter("lane"):
        for tag in ("predecessor", "successor"):
            etree.SubElement(lane.find("link"), tag, id=lane.get("id"))
    with pytest.warns(UserWarning, match="Dropped 6 lanes"):
        m = MapOdr.create(etree.tostring(tree, encoding="unicode"), "short_section", step_size=1.0, parse_map=True)
    assert {idx.section_id for idx in m.lanes} == {0, 2}
    assert all(lane.centerline.length > 0 for lane in m.lanes.values())
    assert {b.section_id for b in m.lane_boundaries} == {0, 2}
    for idx, lane in m.lanes.items():
        other = idx._replace(section_id=2 - idx.section_id)
        assert (lane.successor_ids if idx.section_id == 0 else lane.predecessor_ids) == [other]


def drop_phantoms(links):
    # lane "F" has no geometry; links maps a lane to its (predecessor_ids, successor_ids)
    lanes = {
        idx: SimpleNamespace(
            idx=idx,
            centerline=LineString() if idx == "F" else LineString([(0, 0), (1, 0)]),
            predecessor_ids=list(pred),
            successor_ids=list(succ),
            left_boundary_id=None,
            right_boundary_id=None,
        )
        for idx, (pred, succ) in links.items()
    }
    m = SimpleNamespace(_lanes=lanes, _lane_boundaries={}, name="stub", step_size=1.0)
    with pytest.warns(UserWarning, match="Dropped 1 lanes"):
        MapOdr._drop_phantom_lanes(m)
    return {idx: (lane.predecessor_ids, lane.successor_ids) for idx, lane in lanes.items()}


def test_phantom_bridging_keeps_the_orientation_of_an_end_to_end_contact():
    # B's end touches the phantom's end, like a road linked with contactPoint="end" (example_files/soderleden.xodr)
    links = drop_phantoms({"A": ([], ["F"]), "F": (["A"], ["B"]), "B": ([], ["F"])})
    assert links == {"A": ([], ["B"]), "B": ([], ["A"])}


def test_phantom_bridging_keeps_the_orientation_of_a_start_to_start_contact():
    links = drop_phantoms({"A": (["F"], []), "F": (["A"], ["B"]), "B": (["F"], [])})
    assert links == {"A": (["B"], []), "B": (["A"], [])}


def test_phantom_bridging_of_one_sided_links():
    # only the phantom lists A, only B lists the phantom
    links = drop_phantoms({"A": ([], []), "F": (["A"], []), "B": (["F"], [])})
    assert links == {"A": ([], ["B"]), "B": (["A"], [])}
