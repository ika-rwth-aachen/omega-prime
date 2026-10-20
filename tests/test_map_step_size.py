import copy
from pathlib import Path

import pytest
from lxml import etree

import omega_prime
from omega_prime.map_odr import MapOdr

FILES = Path(__file__).parent / "files"


@pytest.fixture(scope="module")
def cut_in_with_map():
    return omega_prime.Recording.from_file(str(FILES / "alks_cut-in.osi"), str(FILES / "straight_500m.xodr"))


@pytest.mark.parametrize("suffix", [".mcap", ".parquet"])
def test_step_size_reaches_the_embedded_map(cut_in_with_map, tmp_path, suffix):
    path = tmp_path / f"cut_in{suffix}"
    cut_in_with_map.to_file(path)
    default = omega_prime.Recording.from_file(path)
    coarse = omega_prime.Recording.from_file(path, step_size=1.0)
    assert isinstance(coarse.map, MapOdr)
    assert default.map.step_size == 0.01
    assert coarse.map.step_size == 1.0
    assert max(len(lane.centerline.coords) for lane in coarse.map.lanes.values()) < max(
        len(lane.centerline.coords) for lane in default.map.lanes.values()
    )


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
