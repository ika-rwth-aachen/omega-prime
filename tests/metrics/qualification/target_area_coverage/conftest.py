"""."""

import pytest


_EXPECTED_COORDS_LOCAL = [
    (0.0, 0.0),
    (10.0, 0.0),
    (10.0, 10.0),
    (0.0, 10.0),
]

_PROJ_UTM32N = "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs"
_EXPECTED_COORDS_WGS84 = [
    (6.050307, 50.779819),
    (6.050280, 50.779298),
    (6.050986, 50.779307),
    (6.050927, 50.779789),
]
_EXPECTED_COORDS_UTM32N = [
    (292061.502, 5629488.771),
    (292057.287, 5629430.927),
    (292107.088, 5629429.941),
    (292105.068, 5629483.691),
]


@pytest.fixture
def expected_coords_local() -> list[tuple[float, float]]:
    return _EXPECTED_COORDS_LOCAL.copy()


@pytest.fixture
def proj_utm32n() -> str:
    return _PROJ_UTM32N


@pytest.fixture
def expected_coords_wgs84() -> list[tuple[float, float]]:
    return _EXPECTED_COORDS_WGS84.copy()


@pytest.fixture
def expected_coords_utm32n() -> list[tuple[float, float]]:
    return _EXPECTED_COORDS_UTM32N.copy()
