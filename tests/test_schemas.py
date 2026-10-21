import betterosi
import pandera.errors as pa_errors
import polars as pl
import pytest

from omega_prime.schemas import recording_moving_object_schema, polars_schema


def _make_valid_row(**overrides):
    row = {
        "frame": 0,
        "total_nanos": 0,
        "idx": 1,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "vel_x": 0.0,
        "vel_y": 0.0,
        "vel_z": 0.0,
        "acc_x": 0.0,
        "acc_y": 0.0,
        "acc_z": 0.0,
        "length": 4.5,
        "width": 1.8,
        "height": 1.5,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "type": int(betterosi.MovingObject.Type.VEHICLE),
        "role": int(betterosi.MovingObject.VehicleClassification.Role.CIVIL),
        "subtype": int(betterosi.MovingObject.VehicleClassification.Type.CAR),
    }
    row.update(overrides)
    return row


def test_schema_accepts_osi_3_8_subtypes():
    # Subtypes 18 (MICROMOBILITY_DEVICE) and 22 (LAND_VEHICLE) are valid in OSI 3.8
    micromobility = int(betterosi.MovingObject.VehicleClassification.Type.MICROMOBILITY_DEVICE)
    land_vehicle = int(betterosi.MovingObject.VehicleClassification.Type.LAND_VEHICLE)

    df = pl.DataFrame(
        [
            _make_valid_row(idx=1, total_nanos=0, subtype=micromobility),
            _make_valid_row(idx=2, total_nanos=0, subtype=land_vehicle),
        ],
        schema_overrides=polars_schema,
    )

    validated = recording_moving_object_schema.validate(df)
    assert validated.height == 2


def test_schema_rejects_invalid_subtype():
    df = pl.DataFrame(
        [
            _make_valid_row(idx=1, total_nanos=0, subtype=99),
        ],
        schema_overrides=polars_schema,
    )

    with pytest.raises(pa_errors.SchemaError):
        recording_moving_object_schema.validate(df)


def test_recording_validation_with_osi_3_8_subtypes():
    from omega_prime import Recording

    micromobility = int(betterosi.MovingObject.VehicleClassification.Type.MICROMOBILITY_DEVICE)
    land_vehicle = int(betterosi.MovingObject.VehicleClassification.Type.LAND_VEHICLE)

    df = pl.DataFrame(
        [
            _make_valid_row(idx=1, total_nanos=0, subtype=micromobility),
            _make_valid_row(idx=2, total_nanos=0, subtype=land_vehicle),
        ],
        schema_overrides=polars_schema,
    )

    # Validating recording should not raise SchemaErrors
    rec = Recording(df=df, validate=True)
    assert rec.df.height == 2

