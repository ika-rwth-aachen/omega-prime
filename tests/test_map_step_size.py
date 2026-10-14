from pathlib import Path

import pytest

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
