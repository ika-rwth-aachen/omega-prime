import math

import betterosi
import polars as pl
import pyproj
import pytest
from lxml import etree
from typer.testing import CliRunner

from omega_prime import MapOdr, Recording
from omega_prime.cli import app
from omega_prime.map import ProjectionOffset


SOURCE_PROJECTION = "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs"
TARGET_PROJECTION = "+proj=tmerc +lat_0=50 +lon_0=6 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

OPEN_DRIVE = f"""<OpenDRIVE>
  <header>
    <geoReference><![CDATA[{TARGET_PROJECTION}]]></geoReference>
    <offset x="100" y="200" z="5" hdg="0.25"/>
  </header>
  <road id="1">
    <elevationProfile><elevation s="0" a="10" b="1" c="2" d="3"/></elevationProfile>
    <lateralProfile>
      <superelevation s="0" a="1" b="2" c="3" d="4"/>
      <crossfall s="0" side="both" a="1" b="2" c="3" d="4"/>
      <shape s="0" t="0" a="1" b="2" c="3" d="4"/>
    </lateralProfile>
  </road>
</OpenDRIVE>"""


def make_recording() -> Recording:
    recording = Recording(
        [
            {
                "total_nanos": 1_000_000_000,
                "idx": 7,
                "x": 12.0,
                "y": 34.0,
                "z": 1.5,
                "vel_x": 0.0,
                "vel_y": 0.0,
                "vel_z": 0.0,
                "acc_x": 0.0,
                "acc_y": 0.0,
                "acc_z": 0.0,
                "length": 4.0,
                "width": 2.0,
                "height": 1.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.5,
                "type": int(betterosi.MovingObject.Type.VEHICLE),
                "role": 0,
                "subtype": 0,
            }
        ],
        map=MapOdr.create(OPEN_DRIVE, name="test"),
        projections={
            "proj_string": SOURCE_PROJECTION,
            None: ProjectionOffset(x=1.0, y=2.0, z=3.0, yaw=0.0),
        },
    )
    recording._df = recording._df.with_columns(
        pl.lit(1012.0).alias("x_original"),
        pl.lit(2034.0).alias("y_original"),
        pl.lit(6.5).alias("z_original"),
    )
    return recording


def test_materialized_osi_export_uses_current_coordinates_and_map_projection():
    recording = make_recording()

    normal_gt = next(recording.to_osi_gts())
    materialized_gt = next(recording.to_osi_gts(materialize_projection=True))

    assert normal_gt.moving_object[0].base.position.x == 1012.0
    assert normal_gt.proj_string == SOURCE_PROJECTION
    assert normal_gt.proj_frame_offset.position.x == 1.0

    assert materialized_gt.moving_object[0].base.position.x == 12.0
    assert materialized_gt.moving_object[0].base.position.y == 34.0
    assert materialized_gt.proj_string == TARGET_PROJECTION
    assert materialized_gt.proj_frame_offset.position.x == 100.0
    assert materialized_gt.proj_frame_offset.position.y == 200.0
    assert materialized_gt.proj_frame_offset.yaw == 0.25


def test_materialized_parquet_is_idempotent_when_loaded(tmp_path):
    output = tmp_path / "projected.parquet"
    make_recording().to_parquet(output, materialize_projection=True)

    reloaded = Recording.from_file(output, validate=False)

    assert reloaded.df["x"].item() == pytest.approx(12.0)
    assert reloaded.df["y"].item() == pytest.approx(34.0)
    assert reloaded.df["z"].item() == pytest.approx(1.5)


@pytest.mark.parametrize("apply_proj", [True, False])
def test_parquet_with_embedded_map_respects_apply_proj(tmp_path, apply_proj):
    source = tmp_path / "input.parquet"
    make_recording().to_parquet(source)
    expected = Recording.from_parquet(source)
    if apply_proj:
        expected.apply_projections()

    reloaded = Recording.from_file(source, apply_proj=apply_proj)

    assert reloaded.df.select("x", "y", "z").equals(expected.df.select("x", "y", "z"))
    assert ("x_original" in reloaded.df.columns) == apply_proj


@pytest.mark.parametrize("suffix", [".parquet", ".mcap"])
def test_materialize_projection_cli_with_embedded_parquet_map(tmp_path, suffix):
    source = tmp_path / "input.parquet"
    output = tmp_path / f"output{suffix}"
    make_recording().to_parquet(source)
    expected = Recording.from_parquet(source).apply_projections()

    result = CliRunner().invoke(app, ["materialize-projection", str(source), str(output), "--flatten-map"])

    assert result.exit_code == 0, result.exception
    reloaded = Recording.from_file(output)
    for coordinate in ("x", "y", "z"):
        assert reloaded.df[coordinate].item() == pytest.approx(expected.df[coordinate].item())


def test_materialization_requires_an_applied_projection():
    recording = make_recording()
    recording._df = recording._df.drop("x_original", "y_original", "z_original")

    with pytest.raises(ValueError, match="No applied projection found"):
        next(recording.to_osi_gts(materialize_projection=True))


def test_flatten_elevation_zeros_all_surface_polynomials():
    map_odr = MapOdr.create(OPEN_DRIVE, name="test")
    map_odr.flatten_elevation()
    root = etree.fromstring(map_odr.odr_xml.encode())
    elements = root.xpath(
        "//*[local-name()='elevation' or local-name()='superelevation' "
        "or local-name()='crossfall' or local-name()='shape']"
    )

    assert len(elements) == 4
    assert all(element.get(name) == "0" for element in elements for name in ("a", "b", "c", "d"))
    assert elements[2].get("side") == "both"
    assert elements[3].get("t") == "0"
    assert map_odr._xodr_map is None
    assert map_odr._lanes is None
    assert map_odr._lane_boundaries is None


def test_apply_projections_without_proj_string_raises_value_error():
    recording = Recording(
        [
            {
                "total_nanos": 1_000_000_000,
                "idx": 1,
                "x": 10.0,
                "y": 20.0,
                "z": 1.0,
                "vel_x": 0.0,
                "vel_y": 0.0,
                "vel_z": 0.0,
                "acc_x": 0.0,
                "acc_y": 0.0,
                "acc_z": 0.0,
                "length": 4.0,
                "width": 2.0,
                "height": 1.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "type": int(betterosi.MovingObject.Type.VEHICLE),
                "role": 0,
                "subtype": 0,
            }
        ],
        projections={None: ProjectionOffset(x=1.0, y=2.0, z=0.0, yaw=0.0)},
    )
    with pytest.raises(ValueError, match="No proj_string information available"):
        recording.apply_projections()


# --- directions under projection ----------------------------------------------------------------------------------
# Real-world setup: object data in a UTM-aligned local frame, map in a local transverse Mercator projection whose
# central meridian (6.05 deg) differs from the UTM zone meridian (9 deg) -> the grids are rotated by about 2.29 deg.
UTM_32N = "EPSG:32632"
LOCAL_TMERC = (
    "+proj=tmerc +lat_0=50.7795585854429 +lon_0=6.05069819822389 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
UTM_ORIGIN = ProjectionOffset(x=292087.91958096693, y=5629458.721799827, z=255.0, yaw=0.0)


def open_drive(proj_string: str, offset: str = "") -> str:
    return f"""<OpenDRIVE>
  <header>
    <geoReference><![CDATA[{proj_string}]]></geoReference>
    {offset}
  </header>
</OpenDRIVE>"""


def moving_rows(yaw: float, speed: float = 10.0, accel: float = 1.0, n: int = 3) -> list[dict]:
    "An object driving along its heading, sampled at 10 Hz."
    rows = []
    for k in range(n):
        s = speed * 0.1 * k
        rows.append(
            {
                "total_nanos": 1_000_000_000 + k * 100_000_000,
                "idx": 3,
                "x": 20.0 + s * math.cos(yaw),
                "y": -15.0 + s * math.sin(yaw),
                "z": 0.5,
                "vel_x": speed * math.cos(yaw),
                "vel_y": speed * math.sin(yaw),
                "vel_z": 0.0,
                "acc_x": accel * math.cos(yaw),
                "acc_y": accel * math.sin(yaw),
                "acc_z": 0.0,
                "length": 18.0,
                "width": 2.5,
                "height": 3.2,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": yaw,
                "type": int(betterosi.MovingObject.Type.VEHICLE),
                "role": 0,
                "subtype": 0,
            }
        )
    return rows


def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def assert_directions_consistent(df: pl.DataFrame, tol: float = 1e-6):
    "yaw, velocity, acceleration, polygon axis and the motion between projected positions point the same way."
    df = df.sort("total_nanos")
    yaw = df["yaw"].to_numpy()
    motion = math.atan2(df["y"][1] - df["y"][0], df["x"][1] - df["x"][0])
    assert abs(angle_diff(yaw[0], motion)) < 1e-5
    for row in df.iter_rows(named=True):
        assert abs(angle_diff(math.atan2(row["vel_y"], row["vel_x"]), row["yaw"])) < tol
        assert abs(angle_diff(math.atan2(row["acc_y"], row["acc_x"]), row["yaw"])) < tol
        assert math.hypot(row["vel_x"], row["vel_y"]) == pytest.approx(10.0)
        corners = list(row["polygon"].exterior.coords)
        front_mid = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
        axis = math.atan2(front_mid[1] - row["y"], front_mid[0] - row["x"])
        assert abs(angle_diff(axis, row["yaw"])) < 1e-6


def test_apply_projections_rotates_directions_by_grid_convergence():
    yaw0 = 0.3
    recording = Recording(
        moving_rows(yaw0),
        map=MapOdr.create(open_drive(LOCAL_TMERC), name="tmerc"),
        projections={"proj_string": UTM_32N, None: UTM_ORIGIN},
    )

    recording.apply_projections()
    df = recording.df

    transformer = pyproj.Transformer.from_crs(UTM_32N, LOCAL_TMERC, always_xy=True)
    x0, y0 = transformer.transform(UTM_ORIGIN.x, UTM_ORIGIN.y)
    x1, y1 = transformer.transform(UTM_ORIGIN.x + 100.0, UTM_ORIGIN.y)
    convergence = math.atan2(y1 - y0, x1 - x0)
    # UTM grid east, seen in the map frame (the map's x axis is rotated by -2.2857 deg against UTM grid east)
    assert math.degrees(convergence) == pytest.approx(2.2857, abs=1e-3)

    for yaw in df["yaw"]:
        assert angle_diff(yaw, yaw0 + convergence) == pytest.approx(0.0, abs=1e-5)
    assert_directions_consistent(df)
    assert df["yaw_original"].to_list() == pytest.approx([yaw0] * 3)
    assert df["vel_x_original"].to_list() == pytest.approx([10.0 * math.cos(yaw0)] * 3)


def test_apply_projections_rotates_directions_by_offset_yaw():
    yaw0 = -1.0
    offset_yaw = math.radians(30.0)
    recording = Recording(
        moving_rows(yaw0),
        map=MapOdr.create(open_drive(LOCAL_TMERC), name="tmerc"),
        projections={"proj_string": LOCAL_TMERC, None: ProjectionOffset(x=5.0, y=-3.0, z=0.0, yaw=offset_yaw)},
    )

    recording.apply_projections()
    df = recording.df

    for yaw in df["yaw"]:
        assert angle_diff(yaw, yaw0 + offset_yaw) == pytest.approx(0.0, abs=1e-9)
    assert_directions_consistent(df, tol=1e-9)


def test_apply_projections_keeps_directions_consistent_with_map_offset_heading():
    recording = Recording(
        moving_rows(2.5),
        map=MapOdr.create(OPEN_DRIVE, name="test"),
        projections={"proj_string": SOURCE_PROJECTION, None: ProjectionOffset(x=1.0, y=2.0, z=3.0, yaw=0.1)},
    )

    recording.apply_projections()

    assert_directions_consistent(recording.df)


def test_export_after_projection_restores_original_directions(tmp_path):
    rows = moving_rows(0.7)
    recording = Recording(
        rows,
        map=MapOdr.create(open_drive(LOCAL_TMERC), name="tmerc"),
        projections={"proj_string": UTM_32N, None: UTM_ORIGIN},
    )
    recording.apply_projections()

    gt = next(recording.to_osi_gts())
    base = gt.moving_object[0].base
    assert base.orientation.yaw == pytest.approx(rows[0]["yaw"])
    assert base.velocity.x == pytest.approx(rows[0]["vel_x"])
    assert base.velocity.y == pytest.approx(rows[0]["vel_y"])
    assert base.acceleration.x == pytest.approx(rows[0]["acc_x"])
    assert base.position.x == pytest.approx(rows[0]["x"])

    output = tmp_path / "roundtrip.parquet"
    recording.to_parquet(output)
    reloaded = Recording.from_parquet(output)
    for column in ("x", "y", "yaw", "vel_x", "vel_y", "acc_x", "acc_y"):
        assert reloaded.df.sort("total_nanos")[column].to_list() == pytest.approx([r[column] for r in rows])
    assert not any(c.endswith("_original") for c in reloaded.df.columns)


def test_materialized_export_keeps_projected_directions(tmp_path):
    recording = Recording(
        moving_rows(0.7),
        map=MapOdr.create(open_drive(LOCAL_TMERC), name="tmerc"),
        projections={"proj_string": UTM_32N, None: UTM_ORIGIN},
    )
    recording.apply_projections()
    projected = recording.df.sort("total_nanos")
    output = tmp_path / "materialized.parquet"
    recording.to_parquet(output, materialize_projection=True)

    reloaded = Recording.from_file(output, validate=False).df.sort("total_nanos")

    for column in ("x", "y", "yaw", "vel_x", "vel_y"):
        assert reloaded[column].to_list() == pytest.approx(projected[column].to_list(), abs=1e-6)
    assert_directions_consistent(reloaded)


def test_apply_projections_warns_about_frames_without_offset():
    rows = moving_rows(0.0)
    recording = Recording(
        rows,
        map=MapOdr.create(open_drive(LOCAL_TMERC), name="tmerc"),
        projections={"proj_string": UTM_32N, rows[0]["total_nanos"]: UTM_ORIGIN, rows[1]["total_nanos"]: None},
    )

    with pytest.warns(UserWarning, match="2 of 3 rows have no projection offset"):
        recording.apply_projections()


def test_apply_projections_uses_zero_offset_for_explicit_none_frame():
    rows = moving_rows(0.0)
    recording = Recording(
        rows,
        map=MapOdr.create(open_drive(LOCAL_TMERC), name="tmerc"),
        projections={
            "proj_string": LOCAL_TMERC,
            None: ProjectionOffset(x=100.0, y=50.0, z=0.0, yaw=0.0),
            rows[1]["total_nanos"]: None,
        },
    )

    recording.apply_projections()
    df = recording.df.sort("total_nanos")

    # The frame with an explicit None is not shifted, matching the export, which writes no offset for it.
    assert df["x"][1] == pytest.approx(rows[1]["x"], abs=1e-6)
    assert df["x"][0] == pytest.approx(rows[0]["x"] + 100.0, abs=1e-6)
    assert recording._projection_for_timestamp(rows[1]["total_nanos"])[1] is None


def test_apply_projections_rotates_motion_independent_of_yaw():
    # The heading differs from the direction of motion (e.g. a reversing or drifting vehicle).
    rows = moving_rows(0.3)
    yaw0 = 0.3 + 2.0
    for row in rows:
        row["yaw"] = yaw0
    recording = Recording(
        rows,
        map=MapOdr.create(open_drive(LOCAL_TMERC), name="tmerc"),
        projections={"proj_string": UTM_32N, None: UTM_ORIGIN},
    )

    recording.apply_projections()
    df = recording.df.sort("total_nanos")

    motion = math.atan2(df["y"][1] - df["y"][0], df["x"][1] - df["x"][0])
    for row in df.iter_rows(named=True):
        assert angle_diff(row["yaw"], motion + 2.0) == pytest.approx(0.0, abs=1e-5)
        assert angle_diff(math.atan2(row["vel_y"], row["vel_x"]), motion) == pytest.approx(0.0, abs=1e-5)
        assert angle_diff(math.atan2(row["acc_y"], row["acc_x"]), motion) == pytest.approx(0.0, abs=1e-5)


def test_apply_projections_adds_missing_direction_originals():
    rows = moving_rows(0.7)
    for row in rows:
        row.update(x_original=row["x"], y_original=row["y"], z_original=row["z"])
    recording = Recording(
        rows,
        map=MapOdr.create(open_drive(LOCAL_TMERC), name="tmerc"),
        projections={"proj_string": UTM_32N, None: UTM_ORIGIN},
    )
    assert "yaw_original" not in recording.df.columns

    recording.apply_projections()

    df = recording.df.sort("total_nanos")
    assert df["yaw_original"].to_list() == pytest.approx([0.7] * 3)
    base = next(recording.to_osi_gts()).moving_object[0].base
    assert base.orientation.yaw == pytest.approx(rows[0]["yaw"])
    assert base.velocity.x == pytest.approx(rows[0]["vel_x"])


def test_frame_numbers_follow_time():
    rows = list(reversed(moving_rows(0.0, n=5)))
    recording = Recording(rows)

    frames = recording.df.sort("total_nanos")["frame"].to_list()

    assert frames == [0, 1, 2, 3, 4]


def test_from_file_skips_projection_without_proj_string(tmp_path):
    source = tmp_path / "no_proj.parquet"
    recording = Recording(
        [
            {
                "total_nanos": 1_000_000_000,
                "idx": 1,
                "x": 10.0,
                "y": 20.0,
                "z": 1.0,
                "vel_x": 0.0,
                "vel_y": 0.0,
                "vel_z": 0.0,
                "acc_x": 0.0,
                "acc_y": 0.0,
                "acc_z": 0.0,
                "length": 4.0,
                "width": 2.0,
                "height": 1.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "type": int(betterosi.MovingObject.Type.VEHICLE),
                "role": 0,
                "subtype": 0,
            }
        ],
    )
    recording.to_parquet(source)
    reloaded = Recording.from_file(source, apply_proj=True)
    assert reloaded.df["x"].item() == pytest.approx(10.0)
    assert "x_original" not in reloaded.df.columns
