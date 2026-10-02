import json
from pathlib import Path

import betterosi
import numpy as np
import pyarrow.parquet as pq
import pytest

import omega_prime
from omega_prime.map import ProjectionOffset
from omega_prime.recording import PRECIPITATION_AMOUNT_REFERENCE_TYPE, PRECIPITATION_INTENSITY_REFERENCE_TYPE

p = Path(__file__).parent.parent / "example_files/"
with open(p / "mapping.json") as f:
    mapping = json.load(f)

EC = betterosi.EnvironmentalConditions
Precipitation = betterosi.EnvironmentalConditionsPrecipitation
Fog = betterosi.EnvironmentalConditionsFog

MODERATE = EC(precipitation=Precipitation.MODERATE, fog=Fog.GOOD_VISIBILITY, temperature=283.15)
HEAVY = EC(
    precipitation=Precipitation.HEAVY,
    fog=Fog.MIST,
    wind=betterosi.EnvironmentalConditionsWind(origin_direction=1.5, speed=0.0),
)


def load(i: int = 3, **kwargs) -> omega_prime.Recording:
    return omega_prime.Recording.from_file(p / mapping[i][0], p / mapping[i][1], validate=False, **kwargs)


def set_time_varying(rec: omega_prime.Recording) -> list[int]:
    "Set MODERATE on the first half and HEAVY on the second half of the recording. Returns the sorted timestamps."
    nanos = sorted(rec.nanos2frame)
    rec.set_environmental_conditions(MODERATE)
    rec.set_environmental_conditions(HEAVY, nanos=nanos[len(nanos) // 2 :])
    return nanos


def test_mcap_roundtrip_constant(tmp_path):
    rec = load()
    rec.set_environmental_conditions(MODERATE)
    rec.to_mcap(tmp_path / "constant.mcap")

    reloaded = omega_prime.Recording.from_file(tmp_path / "constant.mcap", validate=False)
    assert set(reloaded.environmental_conditions) == set(rec.nanos2frame)
    assert all(ec == MODERATE for ec in reloaded.environmental_conditions.values())


def test_mcap_roundtrip_time_varying(tmp_path):
    rec = load()
    nanos = set_time_varying(rec)
    rec.to_mcap(tmp_path / "varying.mcap")

    reloaded = omega_prime.Recording.from_file(tmp_path / "varying.mcap", validate=False)
    assert reloaded.environmental_conditions[nanos[0]] == MODERATE
    assert reloaded.environmental_conditions[nanos[-1]] == HEAVY
    assert reloaded.environmental_conditions == rec.environmental_conditions


def test_load_save_preserves_conditions_of_input(tmp_path):
    rec = load()
    set_time_varying(rec)
    rec.to_mcap(tmp_path / "input.mcap")

    # conditions are already part of the input file and are not touched in between
    omega_prime.Recording.from_file(tmp_path / "input.mcap", validate=False).to_mcap(tmp_path / "output.mcap")

    reloaded = omega_prime.Recording.from_file(tmp_path / "output.mcap", validate=False)
    assert reloaded.environmental_conditions == rec.environmental_conditions


def test_absent_conditions_stay_absent(tmp_path):
    rec = load()
    assert rec.environmental_conditions == {}
    assert all(gt.environmental_conditions is None for gt in rec.to_osi_gts())

    rec.to_mcap(tmp_path / "absent.mcap")
    gts = betterosi.read(tmp_path / "absent.mcap", return_ground_truth=True, mcap_return_betterosi=True)
    assert all(gt.environmental_conditions is None for gt in gts)
    assert omega_prime.Recording.from_file(tmp_path / "absent.mcap", validate=False).environmental_conditions == {}


def test_parquet_roundtrip(tmp_path):
    rec = load()
    nanos = sorted(rec.nanos2frame)
    # MODERATE on the first quarter, nothing on the second quarter, HEAVY on the second half
    rec.set_environmental_conditions(MODERATE, nanos=nanos[: len(nanos) // 4])
    rec.set_environmental_conditions(HEAVY, nanos=nanos[len(nanos) // 2 :])
    rec.to_parquet(tmp_path / "conditions.parquet")

    metadata = pq.read_schema(tmp_path / "conditions.parquet").metadata
    runs = json.loads(metadata[b"environmental_conditions_json"])["runs"]
    assert [(r["start"], r["end"]) for r in runs] == [
        (nanos[0], nanos[len(nanos) // 4 - 1]),
        (nanos[len(nanos) // 2], nanos[-1]),
    ]

    reloaded = omega_prime.Recording.from_file(tmp_path / "conditions.parquet", validate=False)
    assert reloaded.environmental_conditions == rec.environmental_conditions
    assert nanos[len(nanos) // 4] not in reloaded.environmental_conditions


def test_parquet_without_conditions(tmp_path):
    rec = load()
    rec.to_parquet(tmp_path / "absent.parquet")
    assert b"environmental_conditions_json" not in pq.read_schema(tmp_path / "absent.parquet").metadata
    assert omega_prime.Recording.from_file(tmp_path / "absent.parquet", validate=False).environmental_conditions == {}


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


def test_interpolate_keeps_traffic_light_states():
    def traffic_light(color):
        return [
            betterosi.TrafficLight(
                id=betterosi.Identifier(value=1),
                classification=betterosi.TrafficLightClassification(color=color),
            )
        ]

    red = traffic_light(betterosi.TrafficLightClassificationColor.RED)
    green = traffic_light(betterosi.TrafficLightClassificationColor.GREEN)

    rec = load()
    nanos = sorted(rec.nanos2frame)
    rec.traffic_light_states = {n: red if i < len(nanos) // 2 else green for i, n in enumerate(nanos)}

    rec.interpolate(hz=10)

    new_nanos = sorted(rec.nanos2frame)
    assert set(rec.traffic_light_states) == set(new_nanos)
    assert rec.traffic_light_states[new_nanos[0]] == red
    assert rec.traffic_light_states[new_nanos[-1]] == green


def test_interpolate_keeps_projections(tmp_path):
    rec = load(1, apply_proj=False)
    proj_string = rec.projections["proj_string"]
    assert proj_string is not None
    # the projection of the example file is an identity, use an offset to make the projection matter
    offset = ProjectionOffset(x=100.0, y=50.0, z=2.0, yaw=0.1)
    rec.projections = {"proj_string": proj_string, **{n: offset for n in rec.nanos2frame}}
    rec.apply_projections()
    assert not np.allclose(rec.df["x"], rec.df["x_original"])

    rec.interpolate(hz=10)

    new_nanos = set(rec.nanos2frame)
    assert rec.projections["proj_string"] == proj_string
    assert {k for k in rec.projections if k not in ("proj_string", None)} == new_nanos
    assert all(rec.projections[n] == offset for n in new_nanos)
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


def test_environmental_conditions_df():
    rec = load()
    empty = rec.environmental_conditions_df
    assert empty.height == 0
    assert "precipitation" in empty.columns

    nanos = set_time_varying(rec)
    df = rec.environmental_conditions_df
    assert df.height == len(nanos)
    assert df["total_nanos"].to_list() == nanos
    assert df["frame"].to_list() == [rec.nanos2frame[n] for n in nanos]
    assert df.columns == empty.columns

    first, last = df.row(0, named=True), df.row(-1, named=True)
    assert first["precipitation"] == Precipitation.MODERATE.value
    assert first["temperature"] == pytest.approx(283.15)
    assert first["wind_speed"] is None
    assert last["precipitation"] == Precipitation.HEAVY.value
    assert last["temperature"] is None
    assert last["wind_speed"] == 0.0
    assert last["wind_origin_direction"] == pytest.approx(1.5)
    assert last["sun_azimuth"] is None

    joined = rec.df.join(df, on="total_nanos", how="left")
    assert joined.height == rec.df.height
    assert joined["precipitation"].null_count() == 0


@pytest.mark.parametrize("suffix", [".mcap", ".parquet"])
def test_measured_precipitation_in_source_reference(tmp_path, suffix):
    def reference(reference_type, value):
        return betterosi.ExternalReference(reference="weather_station", type=reference_type, identifier=[value])

    measured = EC(
        precipitation=Precipitation.VERY_LIGHT,
        source_reference=[
            reference(PRECIPITATION_INTENSITY_REFERENCE_TYPE, "0.3"),
            reference(PRECIPITATION_AMOUNT_REFERENCE_TYPE, "0.01"),
        ],
    )
    not_a_number = EC(source_reference=[reference(PRECIPITATION_INTENSITY_REFERENCE_TYPE, "n/a")])

    rec = load()
    nanos = sorted(rec.nanos2frame)
    rec.set_environmental_conditions(measured)
    rec.set_environmental_conditions(not_a_number, nanos=nanos[-2:-1])
    rec.set_environmental_conditions(MODERATE, nanos=nanos[-1:])
    if suffix == ".mcap":
        rec.to_mcap(tmp_path / "measured.mcap")
    else:
        rec.to_parquet(tmp_path / "measured.parquet")

    reloaded = omega_prime.Recording.from_file(tmp_path / f"measured{suffix}", validate=False)
    assert reloaded.environmental_conditions == rec.environmental_conditions

    df = reloaded.environmental_conditions_df
    first = df.row(0, named=True)
    assert first["precipitation"] == Precipitation.VERY_LIGHT.value
    assert first["precipitation_intensity"] == pytest.approx(0.3)
    assert first["precipitation_amount"] == pytest.approx(0.01)
    assert df.row(-2, named=True)["precipitation_intensity"] is None
    assert df.row(-1, named=True)["precipitation_intensity"] is None
    assert df.row(-1, named=True)["precipitation_amount"] is None
