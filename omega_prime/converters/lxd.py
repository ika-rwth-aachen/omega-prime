from .converter import DatasetConverter, NANOS_PER_SEC
from ..recording import Recording
from ..map_odr import MapOdr

from pathlib import Path

import betterosi
import numpy as np
from lxd_io import Dataset
import polars as pl

vct = betterosi.MovingObjectVehicleClassificationType
vehicles = {
    "Car": vct.TYPE_CAR,
    "car": vct.TYPE_CAR,
    "Truck": vct.TYPE_HEAVY_TRUCK,
    "truck_bus": vct.TYPE_HEAVY_TRUCK,
    "truck": vct.TYPE_HEAVY_TRUCK,
    "bicycle": vct.TYPE_BICYCLE,
    "van": vct.TYPE_DELIVERY_VAN,
}
pedestrians = {"pedestrian": betterosi.MovingObjectType.TYPE_PEDESTRIAN}

class lxdConverter(DatasetConverter):
    def __init__(self, dataset_path: str, out_path: str, n_workers=1) -> None:
        self._dataset = Dataset(dataset_path)
        super().__init__(dataset_path, out_path, n_workers=n_workers)

    def get_source_recordings(self):
        return [self._dataset.get_recording(recording_id) for recording_id in self._dataset.recording_ids]

    def get_recordings(self, source_recording):
        yield source_recording

    def get_recording_id(self, recording) -> int:
        return recording.id

    def to_omega_prime_recording(self, recording) -> Recording:
        dt = 1 / recording.get_meta_data("frameRate")

        meta = recording._tracks_meta_data
        meta = meta.with_columns(
            pl.col("class")
            .map_elements(
                (
                    lambda x: betterosi.MovingObjectType.TYPE_VEHICLE
                    if x in vehicles
                    else betterosi.MovingObjectType.TYPE_PEDESTRIAN
                ),
                return_dtype=int,
            )
            .alias("type"),
            pl.col("class")
            .map_elements(
                (lambda x: betterosi.MovingObjectVehicleClassificationRole.ROLE_CIVIL if x in vehicles else -1),
                return_dtype=int,
            )
            .alias("role"),
            pl.col("class")
            .map_elements((lambda x: vehicles[x] if x in vehicles else -1), return_dtype=int)
            .alias("subtype"),
        )
        meta = meta.rename({"trackId": "idx"})

        tracks = recording._get_tracks_data()
        tracks = tracks.rename(
            {
                "xCenter": "x",
                "yCenter": "y",
                "xVelocity": "vel_x",
                "yVelocity": "vel_y",
                "xAcceleration": "acc_x",
                "yAcceleration": "acc_y",
                "trackId": "idx",
                # "width": "width",
                # "length": "length",
            }
        )
        tracks = tracks.join(meta.select(["idx", "role", "type", "subtype"]), on="idx", how="left")
        is_vehicle = pl.col("type") == betterosi.MovingObjectType.TYPE_VEHICLE
        is_bicycle = pl.col("subtype") == betterosi.MovingObjectVehicleClassificationType.TYPE_BICYCLE
        is_pedestrian = pl.col("type") == betterosi.MovingObjectType.TYPE_PEDESTRIAN
        tracks = tracks.with_columns(
            [pl.lit(0.0).alias(k) for k in ["acc_z", "z", "vel_z", "roll", "pitch"]]
            + [
                (((pl.col("heading") + np.pi) % (2 * np.pi)) - np.pi).alias("yaw"),
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
        return Recording(df=tracks, map=MapOdr.from_file(xodr_path), validate=False)
