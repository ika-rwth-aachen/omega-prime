import betterosi
import polars as pl
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
