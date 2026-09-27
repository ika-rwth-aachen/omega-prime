from .converter import DatasetConverter, NANOS_PER_SEC
from ..recording import Recording
from ..map_odr import MapOdr

import betterosi
import numpy as np
from lxd_io import Dataset
import polars as pl
from warnings import warn

vct = betterosi.MovingObject.VehicleClassification.Type
vehicles = {
    "Car": vct.CAR,
    "car": vct.CAR,
    "Truck": vct.HEAVY_TRUCK,
    "truck_bus": vct.BUS,
    "truck": vct.HEAVY_TRUCK,
    "bicycle": vct.BICYCLE,
    "van": vct.DELIVERY_VAN,
    "ad_shuttle": vct.OTHER,
}
pedestrians = {"pedestrian": betterosi.MovingObject.Type.PEDESTRIAN}


class LxdConverter(DatasetConverter):
    def __init__(self, dataset_path: str, out_path: str, n_workers=1) -> None:
        self._dataset = Dataset(dataset_path)
        super().__init__(dataset_path, out_path, n_workers=n_workers)

    def get_source_recordings(self):
        return [self._dataset.get_recording(recording_id) for recording_id in self._dataset.recording_ids]

    def get_recordings(self, source_recording):
        yield source_recording

    def get_recording_name(self, recording) -> str:
        return f"{str(recording.id).zfill(2)}_tracks"

    def to_omega_prime_recording(self, recording) -> Recording:
        dt = 1 / recording.get_meta_data("frameRate")

        meta = recording._tracks_meta_data
        meta = meta.with_columns(
            pl.col("class")
            .map_elements(
                (
                    lambda x: (
                        betterosi.MovingObject.Type.VEHICLE
                        if x in vehicles
                        else betterosi.MovingObject.Type.PEDESTRIAN
                    )
                ),
                return_dtype=int,
            )
            .alias("type"),
            pl.col("class")
            .map_elements(
                (lambda x: betterosi.MovingObject.VehicleClassification.Role.ROLE_CIVIL if x in vehicles else -1),
                return_dtype=int,
            )
            .alias("role"),
            pl.col("class")
            .map_elements((lambda x: vehicles[x] if x in vehicles else -1), return_dtype=int)
            .alias("subtype"),
        )
        if "trackId" in meta.columns:
            meta = meta.rename({"trackId": "idx"})
        else:
            meta = meta.rename({"id": "idx"})

        tracks = recording._get_tracks_data()
        if "trackId" not in tracks.columns:
            tracks = tracks.rename({"id": "trackId"})
        if "xCenter" in tracks.columns:
            tracks = tracks.rename(
                {
                    "xCenter": "x",
                    "yCenter": "y",
                }
            )
        tracks = tracks.rename(
            {
                "xVelocity": "vel_x",
                "yVelocity": "vel_y",
                "xAcceleration": "acc_x",
                "yAcceleration": "acc_y",
                "trackId": "idx",
            }
        )

        if "drivingDirection" in meta.columns:
            # for highD
            highd_attrs = ["drivingDirection"]
        else:
            highd_attrs = []

        tracks = tracks.join(meta.select(["idx", "role", "type", "subtype"] + highd_attrs), on="idx", how="left")

        if "drivingDirection" in meta.columns:
            tracks = tracks.with_columns(
                (pl.col("drivingDirection") * 180).alias("heading"),
                pl.col("width").alias("length"),
                pl.col("height").alias("width"),
            )

        is_vehicle = pl.col("type") == betterosi.MovingObject.Type.VEHICLE
        is_bicycle = pl.col("subtype") == betterosi.MovingObject.VehicleClassificationType.BICYCLE
        is_pedestrian = pl.col("type") == betterosi.MovingObject.Type.PEDESTRIAN
        tracks = tracks.with_columns(
            [pl.lit(0.0).alias(k) for k in ["acc_z", "z", "vel_z", "roll", "pitch"]]
            + [
                ((((pl.col("heading") / 180 * np.pi) + np.pi) % (2 * np.pi)) - np.pi).alias("yaw"),
                (pl.col("frame") * dt * NANOS_PER_SEC).cast(pl.Int64).alias("total_nanos"),
                pl.when(is_vehicle & is_bicycle)
                .then(0.8)
                .when(is_pedestrian)
                .then(0.5)
                .otherwise(pl.col("width"))
                .alias("width"),
                pl.when(is_vehicle & is_bicycle)
                .then(2.0)
                .when(is_pedestrian)
                .then(0.5)
                .otherwise(pl.col("length"))
                .alias("length"),
                pl.when(is_vehicle & is_bicycle).then(1.9).when(is_pedestrian).then(1.8).otherwise(2.0).alias("height"),
            ]
        )

        xodr_path = recording.opendrive_map_file
        if xodr_path is not None:
            map = MapOdr.from_file(xodr_path)
        else:
            map = None
            warn(f"No map associated with recording {recording}")
        return Recording(df=tracks, map=map, validate=False)
