from types import SimpleNamespace
import polars as pl

from omega_prime.locator import Locator


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
