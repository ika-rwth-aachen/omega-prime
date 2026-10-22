import re

import betterosi
import numpy as np
import pytest

import omega_prime
from omega_prime.map import ProjectionOffset
from omega_prime.recording import nearest

from .recording_helpers import EC, HEAVY, MODERATE, ExplicitInitRecording, Fog, Precipitation, load, set_time_varying


class OldInitRecording(omega_prime.Recording):
    "Subclass with the arguments the old `interpolate()` passed to `__init__`."

    def __init__(self, df, map=None, host_vehicle_idx=None):
        super().__init__(df, map=map, host_vehicle_idx=host_vehicle_idx, validate=False)


def test_interpolate_subclass_with_old_init():
    rec = load()
    sub = OldInitRecording(rec.df.drop("polygon"), map=rec.map, host_vehicle_idx=rec.host_vehicle_idx)
    set_time_varying(sub)

    assert sub.interpolate(hz=10) is sub
    assert set(sub.environmental_conditions) == set(sub.nanos2frame)


@pytest.mark.parametrize("suffix", [".mcap", ".parquet"])
def test_interpolate_subclass_without_environmental_conditions_argument(tmp_path, suffix):
    rec = load()
    set_time_varying(rec)
    rec.to_file(tmp_path / f"conditions{suffix}")
    reloaded = ExplicitInitRecording.from_file(tmp_path / f"conditions{suffix}", validate=False)

    reloaded.interpolate(hz=10)

    assert set(reloaded.environmental_conditions) == set(reloaded.nanos2frame)


def test_nearest_matches_the_distance_matrix():
    "`nearest` gives the index the distance matrix of all pairs gives, the earlier candidate on a tie."
    candidates = np.array([0, 10, 20, 40, 41])
    targets = np.array([-5, 0, 4, 5, 6, 15, 30, 40, 41, 100])
    expected = np.abs(candidates - targets[:, None]).argmin(axis=1)
    np.testing.assert_array_equal(nearest(candidates, targets), expected)
    np.testing.assert_array_equal(nearest(np.array([7]), np.array([0, 7, 9])), [0, 0, 0])


def test_interpolate_does_not_share_conditions():
    rec = load()
    rec.set_environmental_conditions(EC(precipitation=Precipitation.LIGHT))

    rec.interpolate(hz=60)

    new_nanos = sorted(rec.nanos2frame)
    rec.environmental_conditions[new_nanos[-1]].fog = Fog.MIST
    assert rec.environmental_conditions[new_nanos[-2]] == EC(precipitation=Precipitation.LIGHT)


def test_interpolate_without_overlapping_timestamps():
    rec = load()
    nanos = sorted(rec.nanos2frame)
    rec.set_environmental_conditions(MODERATE)
    rec.traffic_light_states = {n: [] for n in nanos}

    rec.interpolate(new_nanos=[nanos[-1] + 10**9])

    assert rec.df.height == 0
    assert rec.environmental_conditions == {}
    assert rec.traffic_light_states == {}


def test_interpolate_keeps_environmental_conditions():
    rec = load()
    nanos = set_time_varying(rec)
    switch = nanos[len(nanos) // 2]

    assert rec.interpolate(hz=10) is rec

    new_nanos = sorted(rec.nanos2frame)
    assert new_nanos != nanos
    assert set(rec.environmental_conditions) == set(new_nanos)
    step = int(1e9 / 10)
    for n in new_nanos:
        if n <= switch - step:
            assert rec.environmental_conditions[n] == MODERATE
        elif n >= switch:
            assert rec.environmental_conditions[n] == HEAVY
    assert rec.environmental_conditions[new_nanos[0]] == MODERATE
    assert rec.environmental_conditions[new_nanos[-1]] == HEAVY


def test_interpolate_keeps_sparse_environmental_conditions():
    rec = load()
    nanos = sorted(rec.nanos2frame)
    # conditions on every 7th timestamp of the first half, nothing on the second half
    sparse = nanos[1 : len(nanos) // 2 : 7]
    rec.set_environmental_conditions(MODERATE, nanos=sparse)

    hz = 2
    rec.interpolate(hz=hz)

    new_nanos = np.array(sorted(rec.nanos2frame))
    assert len(new_nanos) < len(sparse)
    assert all(c == MODERATE for c in rec.environmental_conditions.values())
    # the conditions are kept on the new timestamps nearest to the sparse ones ...
    assert set(rec.environmental_conditions) == {int(new_nanos[np.abs(new_nanos - n).argmin()]) for n in sparse}
    # ... and the half without conditions stays without
    assert max(rec.environmental_conditions) <= sparse[-1] + 1e9 / hz / 2


def downsample_with_partial_last_interval(rec: omega_prime.Recording) -> tuple[list[int], float]:
    "Sorted original timestamps and a frequency whose last new timestamp lies before the last original one."
    nanos = sorted(rec.nanos2frame)
    hz = 1e9 / (7 * (nanos[1] - nanos[0]))
    last_new = np.arange(nanos[0], nanos[-1] + 1, 1e9 / hz, dtype=int)[-1]
    assert nanos[-1] - last_new > 1e9 / hz / 4
    return nanos, hz


def test_interpolate_keeps_conditions_of_partial_last_interval():
    rec = load(1)
    nanos, hz = downsample_with_partial_last_interval(rec)
    rec.set_environmental_conditions(HEAVY, nanos=[nanos[-1]])

    rec.interpolate(hz=hz)

    assert rec.environmental_conditions == {max(rec.nanos2frame): HEAVY}


def test_interpolate_partial_last_interval_does_not_overwrite():
    rec = load(1)
    nanos, hz = downsample_with_partial_last_interval(rec)
    last_new = np.arange(nanos[0], nanos[-1] + 1, 1e9 / hz, dtype=int)[-1]
    # conditions on the original timestamp nearest to the last new one are not overwritten by the trailing ones
    rec.set_environmental_conditions(MODERATE, nanos=[min(nanos, key=lambda n: abs(n - last_new))])
    rec.set_environmental_conditions(HEAVY, nanos=[nanos[-1]])

    rec.interpolate(hz=hz)

    assert rec.environmental_conditions == {max(rec.nanos2frame): MODERATE}


def test_interpolate_with_explicit_timestamps_crops_conditions():
    rec = load(1)
    nanos = sorted(rec.nanos2frame)
    rec.set_environmental_conditions(HEAVY, nanos=[nanos[-1]])

    rec.interpolate(new_nanos=nanos[: len(nanos) // 2 : 5])

    assert rec.environmental_conditions == {}


def test_interpolate_keeps_traffic_light_states():
    def traffic_light(color):
        return [
            betterosi.TrafficLight(
                id=betterosi.Identifier(value=1),
                classification=betterosi.TrafficLight.Classification(color=color),
            )
        ]

    red = traffic_light(betterosi.TrafficLight.Classification.Color.RED)
    green = traffic_light(betterosi.TrafficLight.Classification.Color.GREEN)

    rec = load()
    nanos = sorted(rec.nanos2frame)
    rec.traffic_light_states = {n: red if i < len(nanos) // 2 else green for i, n in enumerate(nanos)}

    rec.interpolate(hz=10)

    new_nanos = sorted(rec.nanos2frame)
    assert set(rec.traffic_light_states) == set(new_nanos)
    assert rec.traffic_light_states[new_nanos[0]] == red
    assert rec.traffic_light_states[new_nanos[-1]] == green

    # upsampled timestamps do not share the states
    rec.interpolate(hz=60)
    new_nanos = sorted(rec.nanos2frame)
    rec.traffic_light_states[new_nanos[0]][0].classification.color = betterosi.TrafficLight.Classification.Color.YELLOW
    colors = [rec.traffic_light_states[n][0].classification.color for n in new_nanos[:2]]
    assert colors == [
        betterosi.TrafficLight.Classification.Color.YELLOW,
        betterosi.TrafficLight.Classification.Color.RED,
    ]


@pytest.mark.parametrize("hz", [10, 60])
@pytest.mark.parametrize("varying", [False, True], ids=["constant_offset", "varying_offset"])
def test_interpolate_keeps_projections(tmp_path, varying, hz):
    rec = load(1, apply_proj=False)
    proj_string = rec.projections["proj_string"]
    assert proj_string is not None
    # the projection of the example file is an identity, use an offset to make the projection matter
    offset = ProjectionOffset(x=100.0, y=50.0, z=2.0, yaw=0.1)
    nanos = sorted(rec.nanos2frame)
    if varying:
        # an offset that moves with every timestamp, like the pose of the ego in an ego-local recording
        offsets = {
            n: ProjectionOffset(x=100.0 + 5.0 * i, y=50.0, z=2.0, yaw=0.1 + 0.01 * i) for i, n in enumerate(nanos)
        }
    else:
        offsets = {n: offset for n in nanos}
    rec.projections = {"proj_string": proj_string, **offsets}
    rec.apply_projections()
    assert not np.allclose(rec.df["x"], rec.df["x_original"])

    rec.interpolate(hz=hz)

    new_nanos = set(rec.nanos2frame)
    assert rec.projections["proj_string"] == proj_string
    assert {k for k in rec.projections if k not in ("proj_string", None)} == new_nanos
    assert all(rec.projections[n] in offsets.values() for n in new_nanos)
    assert "x_original" in rec.df.columns

    # the stored pose must stay in the source frame, so that it is not projected twice on reload
    rec.to_mcap(tmp_path / "interpolated.mcap")
    reloaded = omega_prime.Recording.from_file(tmp_path / "interpolated.mcap", validate=False)
    assert reloaded.projections["proj_string"] == proj_string
    expected = rec.df.sort(["total_nanos", "idx"])
    actual = reloaded.df.sort(["total_nanos", "idx"])
    assert actual.height == expected.height
    for c in ["x", "y", "z"]:
        np.testing.assert_allclose(actual[c].to_numpy(), expected[c].to_numpy(), atol=1e-6)


@pytest.mark.parametrize("hz", [10, 2])
@pytest.mark.parametrize("varying", [False, True], ids=["constant_offset", "varying_offset"])
def test_interpolate_keeps_projections_with_crs_transformation(tmp_path, varying, hz):
    # The original pose is interpolated in the source CRS, the projected one in the map CRS. Since the transformation
    # between both is not linear, these do not commute exactly, but the difference over one sampling step is far below
    # any relevant scale.
    source_proj_string = "+proj=tmerc +lat_0=37.35 +lon_0=-122.0 +k=0.9996 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
    map_proj_string = "+proj=tmerc +lat_0=37.35 +lon_0=-122.5 +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
    rec = load(1, apply_proj=False)
    # `parse()` reads the projection of the map from its xodr on every `apply_projections()`
    rec.map.odr_xml, n_replaced = re.subn(
        r"<geoReference>.*?</geoReference>",
        f"<geoReference><![CDATA[{map_proj_string}]]></geoReference>",
        rec.map.odr_xml,
        flags=re.S,
    )
    assert n_replaced == 1
    nanos = sorted(rec.nanos2frame)
    rec.projections = {
        "proj_string": source_proj_string,
        **{
            n: ProjectionOffset(
                x=100.0 + (5.0 * i if varying else 0.0), y=50.0, z=2.0, yaw=0.1 + (0.01 * i if varying else 0.0)
            )
            for i, n in enumerate(nanos)
        },
    }
    rec.apply_projections()
    # the transformation is not an identity
    assert np.abs(rec.df["x"] - rec.df["x_original"]).min() > 1e3

    rec.interpolate(hz=hz)
    rec.to_mcap(tmp_path / "interpolated.mcap")
    reloaded = omega_prime.Recording.from_file(tmp_path / "interpolated.mcap", validate=False, apply_proj=False)
    reloaded.map = rec.map
    reloaded.apply_projections()

    expected = rec.df.sort(["total_nanos", "idx"])
    actual = reloaded.df.sort(["total_nanos", "idx"])
    assert actual.height == expected.height
    for c in ["x", "y", "z"]:
        np.testing.assert_allclose(actual[c].to_numpy(), expected[c].to_numpy(), atol=1e-6)
