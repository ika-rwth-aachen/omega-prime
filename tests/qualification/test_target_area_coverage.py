"""."""

from types import SimpleNamespace

import polars as pl
import pytest
from pyproj import Transformer

from omega_prime.qualification.target_area_coverage import target_area_coverage


EXPECTED_COORDS = [
    (0.0, 0.0),
    (10.0, 0.0),
    (10.0, 10.0),
    (0.0, 10.0),
]
PROJ_MERCATOR = "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

@pytest.fixture()
def target_area_df_pass() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [0.0, 10.0, 0.0, 10.0],
            "y": [0.0, 0.0, 10.0, 10.0],
            "total_nanos": [0, 0, 1, 1],
        }
    )


@pytest.fixture()
def target_area_df_fail() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [20.0, 30.0],
            "y": [20.0, 30.0],
            "total_nanos": [0, 1],
        }
    )

def test_pass(target_area_df_pass) -> None:
    _df, result_dict = target_area_coverage(
        target_area_df_pass,
        expected_area_coords=EXPECTED_COORDS,
        threshold=80.0,
        frame_threshold=100.0,
        expected_area_crs=None,
    )
    summary = result_dict["target_area_coverage"].collect()
    assert summary["target_area_coverage"][0] == pytest.approx(100.0)
    assert summary["status"][0] == "pass"
    assert summary["frame_coverage"][0] == pytest.approx(100.0)
    assert summary["frame_status"][0] == "pass"

    frames = result_dict["target_area_frame_coverage"].collect()
    assert frames["inside_expected_area"].all()


def test_fail(target_area_df_fail) -> None:
    _df, result_dict = target_area_coverage(
        target_area_df_fail,
        expected_area_coords=EXPECTED_COORDS,
        threshold=80.0,
        frame_threshold=50.0,
        expected_area_crs=None,
    )
    summary = result_dict["target_area_coverage"].collect()
    assert summary["target_area_coverage"][0] == pytest.approx(0.0)
    assert summary["status"][0] == "fail"
    assert summary["frame_coverage"][0] == pytest.approx(0.0)
    assert summary["frame_status"][0] == "fail"

    frames = result_dict["target_area_frame_coverage"].collect()
    assert frames["inside_expected_area"].sum() == 0


def test_transform_wgs84_to_dataset_crs() -> None:
    expected_coords_wgs84 = [
        (8.0, 50.0),
        (8.001, 50.0),
        (8.001, 50.001),
        (8.0, 50.001),
    ]
    transformer = Transformer.from_crs("EPSG:4326", PROJ_MERCATOR, always_xy=True)
    lons, lats = zip(*expected_coords_wgs84)
    xs, ys = transformer.transform(lons, lats)
    df = pl.DataFrame(
        {
            "x": [xs[0], xs[1], xs[2], xs[3]],
            "y": [ys[0], ys[1], ys[2], ys[3]],
            "total_nanos": [0, 0, 1, 1],
        }
    )
    recording = SimpleNamespace(
        projections=[
            {
                "proj_string": PROJ_MERCATOR,
            }
        ]
    )
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=expected_coords_wgs84,
        threshold=100.0,
        frame_threshold=100.0,
        expected_area_crs="EPSG:4326",
        recording=recording,
    )
    summary = result_dict["target_area_coverage"].collect()
    assert summary["target_area_coverage"][0] == pytest.approx(100.0)
    assert summary["status"][0] == "pass"
