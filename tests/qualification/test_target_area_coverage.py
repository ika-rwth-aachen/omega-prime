"""."""

import polars as pl
from pyproj import Transformer

from omega_prime.qualification.target_area_coverage import target_area_coverage, TARGET_AREA_COVERAGE

from .conftest import qualification_assert


EXPECTED_COORDS = [
    (0.0, 0.0),
    (10.0, 0.0),
    (10.0, 10.0),
    (0.0, 10.0),
]
PROJ_MERCATOR = "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def test_pass() -> None:
    df = pl.DataFrame(
        {
            "x": [0.0, 10.0, 0.0, 10.0],
            "y": [0.0, 0.0, 10.0, 10.0],
            "total_nanos": [0, 0, 1, 1],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 100.0, True)


def test_fail() -> None:
    df = pl.DataFrame(
        {
            "x": [20.0, 30.0],
            "y": [20.0, 30.0],
            "total_nanos": [0, 1],
        }
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=EXPECTED_COORDS,
        threshold=80.0,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 0.0, False)


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
    ).lazy()
    _df, result_dict = target_area_coverage(
        df,
        expected_area_coords=expected_coords_wgs84,
        threshold=100.0,
        expected_area_crs="EPSG:4326",
        proj_string=PROJ_MERCATOR,
    )
    qualification_assert(result_dict, TARGET_AREA_COVERAGE, 100.0, True)
