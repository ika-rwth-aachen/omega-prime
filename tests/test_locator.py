from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import shapely

from omega_prime.locator import Locator, ShapelyTrajectoryTools
from omega_prime.map_odr import MapOdr


def test_locate_mv_unsorted_and_stationary():
    locator = Locator.__new__(Locator)

    def fake_xys2sts(xy, polygons=None):
        return pl.DataFrame(
            {
                "s": [float(p[0]) for p in xy],
                "t": [float(p[1]) for p in xy],
                "roadlane_id": [1 for _ in xy],
            }
        )

    locator.xys2sts = fake_xys2sts

    # Timestamps are deliberately unsorted, and vehicle stops at (5.0, 0.0)
    df = pl.DataFrame(
        {
            "total_nanos": [200, 0, 100, 150],
            "x": [10.0, 0.0, 5.0, 5.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "polygon": [None, None, None, None],
        }
    )
    mv = SimpleNamespace(_df=df, polygon=None)

    sts = locator.locate_mv(mv)

    # Output rows must match original mv._df row order
    assert list(sts["time"]) == [200, 0, 100, 150]
    assert list(sts["s"]) == [10.0, 0.0, 5.0, 5.0]
    assert list(sts["t"]) == [0.0, 0.0, 0.0, 0.0]


def test_locate_mv_all_moving():
    locator = Locator.__new__(Locator)

    def fake_xys2sts(xy, polygons=None):
        return pl.DataFrame(
            {
                "s": [float(p[0]) for p in xy],
                "t": [float(p[1]) for p in xy],
                "roadlane_id": [1 for _ in xy],
            }
        )

    locator.xys2sts = fake_xys2sts

    # All rows are moving
    df = pl.DataFrame(
        {
            "total_nanos": [0, 100, 200],
            "x": [0.0, 5.0, 10.0],
            "y": [0.0, 0.0, 0.0],
            "polygon": [None, None, None],
        }
    )
    mv = SimpleNamespace(_df=df, polygon=None)

    sts = locator.locate_mv(mv)
    assert list(sts["time"]) == [0, 100, 200]
    assert list(sts["s"]) == [0.0, 5.0, 10.0]


def straight_map():
    return MapOdr.from_file(Path(__file__).parent / "files" / "straight_500m.xodr", parse_map=True, step_size=1.0)


def test_lane_without_a_direction_is_not_located_on():
    m = straight_map()
    lanes = list(m.lanes.values())
    lanes[0].centerline = shapely.Point(0.0, 0.0)
    with pytest.warns(UserWarning, match="Skipped 1 lanes without a centerline direction"):
        locator = Locator.from_map(m)
    assert [lane.idx for lane in locator.all_lanes] == [lane.idx for lane in lanes[1:]]
    assert len(locator.extended_centerlines) == len(locator.internal2external_laneid) == len(lanes) - 1
    # positions on the remaining lanes are still located
    center = lanes[1].polygon.representative_point()
    sts = locator.xys2sts(np.array([[center.x, center.y]]))
    assert sts["roadlane_id"][0] != lanes[0].idx


def test_locator_without_a_lane_with_a_direction_raises():
    lane = SimpleNamespace(idx=0, centerline=shapely.Point(0.0, 0.0), polygon=None)
    with pytest.warns(UserWarning, match="Skipped 1 lanes"), pytest.raises(ValueError, match="No lane"):
        Locator(all_lanes=[lane])
    with pytest.raises(ValueError, match="No lane"):
        Locator(all_lanes=[])


def test_short_lane_is_extended_along_its_own_geometry():
    # unlike a trajectory below min_length, a lane keeps its own direction however short
    m = straight_map()
    lanes = list(m.lanes.values())
    lanes[0].centerline = shapely.LineString([(0.0, 0.0), (0.04, 0.0)])
    locator = Locator.from_map(m)
    assert len(locator.all_lanes) == len(lanes)
    start, end = np.asarray(locator.extended_centerlines[0].coords)[[0, -1]]
    l_append = ShapelyTrajectoryTools.l_append
    assert np.allclose(start, [-l_append, 0.0]) and np.allclose(end, [0.04 + l_append, 0.0])
