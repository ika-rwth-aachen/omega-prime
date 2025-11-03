import betterosi
import numpy as np
import polars as pl
import pandera.polars as pa
import pandera.extensions as extensions
import polars as pl

pi_valued = pa.Check.between(-np.pi, np.pi)
polars_schema = {
    "total_nanos": pl.Int64,
    "idx": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "vel_x": pl.Float64,
    "vel_y": pl.Float64,
    "vel_z": pl.Float64,
    "acc_x": pl.Float64,
    "acc_y": pl.Float64,
    "acc_z": pl.Float64,
    "length": pl.Float64,
    "width": pl.Float64,
    "height": pl.Float64,
    "roll": pl.Float64,
    "pitch": pl.Float64,
    "yaw": pl.Float64,
    "type": pl.Int64,
    "role": pl.Int64,
    "subtype": pl.Int64,
}


@extensions.register_check_method(
    statistics=["column_name", "column_value", "other_column_name", "other_column_unset_value"]
)
def other_column_set_on_column_value(
    polars_obj, *, column_name: str, column_value, other_column_name: str, other_column_unset_value
):
    return polars_obj.lazyframe.select(
        ~(pl.col(column_name) == column_value).and_(pl.col(other_column_name) == other_column_unset_value)
    )


@extensions.register_check_method(
    statistics=["column_name", "column_value", "other_column_name", "other_column_unset_value"]
)
def other_column_unset_on_column_value(
    polars_obj, *, column_name, column_value, other_column_name: str, other_column_unset_value: int
):
    return polars_obj.lazyframe.select(
        ~(pl.col(column_name) != column_value).and_(pl.col(other_column_name) != other_column_unset_value)
    )


def has_no_frame_skip(df):
    return (
        df.group_by("idx")
        .agg(((pl.col("frame").sort().diff().drop_nulls() == 1).all()).alias("no_skip"))
        .select(pl.col("no_skip").all())
        .row(0)[0]
    )


@extensions.register_check_method()
def check_has_no_frame_skip(polars_obj):
    return polars_obj.lazyframe.select(pl.col("frame").sort().diff().fill_null(1).over("idx") == 1)


recording_moving_object_schema = pa.DataFrameSchema(
    title="DataFrame Schema for ASAM OSI GroundTruth of MovingObjects",
    description="",
    columns={
        "x": pa.Column(polars_schema["x"], title="MovingObject.base.position.x", description="osi3.Vector3d.x"),
        "y": pa.Column(polars_schema["y"], title="MovingObject.base.position.y", description="osi3.Vector3d.y"),
        "z": pa.Column(polars_schema["z"], title="MovingObject.base.position.z", description="osi3.Vector3d.z"),
        "vel_x": pa.Column(polars_schema["vel_x"], title="MovingObject.base.velocity.x", description="osi3.Vector3d.x"),
        "vel_y": pa.Column(polars_schema["vel_y"], title="MovingObject.base.velocity.y", description="osi3.Vector3d.y"),
        "vel_z": pa.Column(polars_schema["vel_z"], title="MovingObject.base.velocity.z", description="osi3.Vector3d.z"),
        "acc_x": pa.Column(
            polars_schema["acc_x"], title="MovingObject.base.acceleration.x", description="osi3.Vector3d.x"
        ),
        "acc_y": pa.Column(
            polars_schema["acc_y"], title="MovingObject.base.acceleration.y", description="osi3.Vector3d.y"
        ),
        "acc_z": pa.Column(
            polars_schema["acc_z"], title="MovingObject.base.acceleration.z", description="osi3.Vector3d.z"
        ),
        "length": pa.Column(
            polars_schema["length"],
            pa.Check.gt(0),
            title="MovingObject.base.dimesion.length",
            description="osi3.Dimenstion3d.length",
        ),
        "width": pa.Column(
            polars_schema["width"],
            pa.Check.gt(0),
            title="MovingObject.base.dimesion.width",
            description="osi3.Dimenstion3d.width",
        ),
        "height": pa.Column(
            polars_schema["height"],
            pa.Check.ge(0),
            title="MovingObject.base.dimesion.height",
            description="osi3.Dimenstion3d.height",
        ),
        "type": pa.Column(
            polars_schema["type"],
            pa.Check.between(
                0, 4, error=f"Type must be one of { ({o.name: o.value for o in betterosi.MovingObjectType}) }"
            ),
            title="MovingObject.type",
            description="osi3.MovingObject.Type",
        ),
        "role": pa.Column(
            polars_schema["role"],
            pa.Check.between(
                -1,
                10,
                error=f"Type must be one of { ({o.name: o.value for o in betterosi.MovingObjectVehicleClassificationRole}) }",
            ),
            title="MovingObject.vehicle_classification.role",
            description="osi3.MovingObject.VehicleClassification.Role",
        ),
        "subtype": pa.Column(
            polars_schema["subtype"],
            pa.Check.between(
                -1,
                17,
                error=f"Subtype must be one of { ({o.name: o.value for o in betterosi.MovingObjectVehicleClassificationType}) }",
            ),
            title="MovingObject.vehicle_classification.type",
            description="osi3.MovingObject.VehicleClassification.Type",
        ),
        "roll": pa.Column(
            polars_schema["roll"],
            pi_valued,
            title="MovingObject.base.orientation.roll",
            description="osi3.Orientation3d.roll",
        ),
        "pitch": pa.Column(
            polars_schema["pitch"],
            pi_valued,
            title="MovingObject.base.orientation.pitch",
            description="osi3.Orientation3d.pitch",
        ),
        "yaw": pa.Column(
            polars_schema["yaw"],
            pi_valued,
            title="MovingObject.base.orientation.yaw",
            description="osi3.Orientation3d.yaw",
        ),
        "idx": pa.Column(
            polars_schema["idx"], pa.Check.ge(0), title="MovingObject.id.value", description="osi3.Identifier.value"
        ),
        "total_nanos": pa.Column(
            polars_schema["total_nanos"],
            pa.Check.ge(0),
            title="GroundTruth.timestamp.nanos+1e9*GroundTruth.timestamp.seconds",
            description="osi3.Timestamp.nanos, osi3.Timestamp.seconds",
        ),
    },
    unique=["idx", "total_nanos"],
    checks=[
        pa.Check.other_column_set_on_column_value(
            "type",
            int(betterosi.MovingObjectType.TYPE_VEHICLE),
            "role",
            -1,
            error="`role` is `-1` despite type beeing `TYPE_VEHICLE`",
        ),
        pa.Check.other_column_unset_on_column_value(
            "type",
            int(betterosi.MovingObjectType.TYPE_VEHICLE),
            "role",
            -1,
            error="`role` is set despite type not beeing `TYPE_VEHICLE`",
        ),
        pa.Check.other_column_set_on_column_value(
            "type",
            int(betterosi.MovingObjectType.TYPE_VEHICLE),
            "subtype",
            -1,
            error="`subtype` is `-1` despite type beeing `TYPE_VEHICLE`",
        ),
        pa.Check.other_column_unset_on_column_value(
            "type",
            int(betterosi.MovingObjectType.TYPE_VEHICLE),
            "subtype",
            -1,
            error="`subtype` is set despite type not beeing `TYPE_VEHICLE`",
        ),
        pa.Check.check_has_no_frame_skip(error="Some objects skip frames during their etistence."),
    ],
)
