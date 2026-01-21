"""
Standalone converter: read perception_msgs/ObjectList messages from ROS 2 bags
and emit omega-prime MCAP files.

The CLI can process specific bag directories or scan a data root for rosbag2
folders (identified via metadata.yaml).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Iterator

import yaml
import numpy as np
import polars as pl
import omega_prime
import betterosi

from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from omega_prime.map import ProjectionOffset

import tf2_ros

import perception_msgs_utils as pmu

# Legacy numpy aliases expected by perception_msgs_utils/tf_transformations
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "maximum_sctype"):

    def _np_maximum_sctype(dtype):
        return np.dtype(np.float64).type

    np.maximum_sctype = _np_maximum_sctype  # type: ignore[attr-defined]

_VCT = betterosi.MovingObjectVehicleClassificationType
_ROLE = betterosi.MovingObjectVehicleClassificationRole
_MOT = betterosi.MovingObjectType


def _class_to_osi(obj) -> tuple[int, int, int]:
    mot = int(_MOT.TYPE_OTHER)
    role = -1
    subtype = -1

    if obj.state.classifications:
        c = pmu.get_class_with_highest_probability(obj)
        ct = int(c.type)
    else:
        ct = 0

    vehicle_map = {
        4: _VCT.TYPE_CAR,
        5: _VCT.TYPE_HEAVY_TRUCK,
        6: _VCT.TYPE_DELIVERY_VAN,
        7: _VCT.TYPE_BUS,
        10: _VCT.TYPE_TRAIN,
        3: _VCT.TYPE_MOTORBIKE,
        2: _VCT.TYPE_BICYCLE,
        11: _VCT.TYPE_TRAILER,
        50: _VCT.TYPE_OTHER,
        51: _VCT.TYPE_OTHER,
        52: _VCT.TYPE_OTHER,
    }

    if ct == 1:
        mot = int(_MOT.TYPE_PEDESTRIAN)
    elif ct == 8:
        mot = int(_MOT.TYPE_ANIMAL)
    elif ct in vehicle_map:
        mot = int(_MOT.TYPE_VEHICLE)
        role = int(_ROLE.ROLE_CIVIL)
        subtype = int(vehicle_map[ct])
    elif ct in (0, 9, 100):
        mot = int(_MOT.TYPE_OTHER)

    return mot, role, subtype


def _object_to_row(obj) -> dict[str, Any]:
    total_nanos = Time.from_msg(obj.state.header.stamp).nanoseconds

    pos = pmu.get_position(obj)
    width = pmu.get_width(obj)
    length = pmu.get_length(obj)
    height = pmu.get_height(obj)

    # pitch and roll might not be available
    try:
        if pmu.index_roll(obj.state.model_id) is not None:
            roll = pmu.get_roll(obj)
    except pmu.UnknownStateEntryError:
        roll = 0.0
    try:
        if pmu.index_pitch(obj.state.model_id) is not None:
            pitch = pmu.get_pitch(obj)
    except pmu.UnknownStateEntryError:
        pitch = 0.0

    yaw = pmu.get_yaw(obj)
    vel_x = pmu.get_vel_x(obj)
    vel_y = pmu.get_vel_y(obj)
    acc_x = pmu.get_acc_x(obj)
    acc_y = pmu.get_acc_y(obj)

    mot, role, subtype = _class_to_osi(obj)

    return {
        "total_nanos": int(total_nanos),
        "idx": int(obj.id),
        "x": float(pos.x),
        "y": float(pos.y),
        "z": float(getattr(pos, "z", 0.0)),
        "vel_x": float(vel_x),
        "vel_y": float(vel_y),
        "vel_z": 0.0,
        "acc_x": float(acc_x),
        "acc_y": float(acc_y),
        "acc_z": 0.0,
        "length": float(length),
        "width": float(width),
        "height": float(height),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
        "type": int(mot),
        "role": int(role),
        "subtype": int(subtype),
    }


def _olist_to_rows(msg) -> list[dict[str, Any]]:
    return [_object_to_row(obj) for obj in msg.objects]


def _load_metadata(bag_dir: Path) -> dict[str, Any]:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found in {bag_dir}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _extract_proj_offset(msg) -> tuple[int, dict[str, Any]]:
    transformation = msg.transforms[0] if hasattr(msg, "transforms") else msg
    translation = transformation.transform.translation
    rotation = transformation.transform.rotation
    ts = int(Time.from_msg(transformation.header.stamp).nanoseconds)
    return ts, {
        "translation": {
            "x": translation.x,
            "y": translation.y,
            "z": translation.z,
        },
        "rotation": {
            "x": rotation.x,
            "y": rotation.y,
            "z": rotation.z,
            "w": rotation.w,
        },
    }


def _yaw_from_quaternion(rotation: dict[str, float]) -> float:
    x = float(rotation.get("x", 0.0))
    y = float(rotation.get("y", 0.0))
    z = float(rotation.get("z", 0.0))
    w = float(rotation.get("w", 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _projection_store_to_projections(
    projection_store: dict[int, dict[str, Any]],
    proj_string: str | None,
) -> dict[int, dict[str, Any]]:
    projections: dict[int, dict[str, Any]] = {}
    # TODO: Handle project string
    if proj_string == "":
        proj_string = None
    for ts, entry in projection_store.items():
        translation = entry.get("translation") or {}
        rotation = entry.get("rotation") or {}
        projections[int(ts)] = {
            "proj_string": proj_string,
            "offset": ProjectionOffset(
                x=float(translation.get("x", 0.0)),
                y=float(translation.get("y", 0.0)),
                z=float(translation.get("z", 0.0)),
                yaw=_yaw_from_quaternion(rotation),
            ),
        }
    return projections


def _storage_id(meta: dict[str, Any]) -> str:
    return meta["rosbag2_bagfile_information"]["storage_identifier"]


def iter_object_list_messages(
    bag_dir: Path,
    topic: str,
    child_frame: str,
    frame: str,
    projection_store: dict[int, dict[str, Any]],
) -> Iterator[Any]:
    metadata = _load_metadata(bag_dir)
    storage_id = _storage_id(metadata)

    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(bag_dir), storage_id=storage_id)
    converter_options = ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    type_map = {info.name: info.type for info in reader.get_all_topics_and_types()}

    if topic not in type_map:
        available = ", ".join(sorted(type_map))
        raise RuntimeError(f"Topic {topic} not found. Available topics: {available}")

    msg_cls = get_message(type_map[topic])
    tf_msg_cls = get_message(type_map["/tf"]) if "/tf" in type_map else None

    while reader.has_next():
        topic_name, data, _ = reader.read_next()
        if topic_name == "/tf" and tf_msg_cls is not None:
            tf_msg = deserialize_message(data, tf_msg_cls)
            for transform in tf_msg.transforms:
                if transform.header.frame_id != frame:
                    continue
                if transform.child_frame_id != child_frame:
                    continue
                ts, proj = _extract_proj_offset(transform)
                projection_store[ts] = proj
            continue
        if topic_name != topic:
            continue
        yield deserialize_message(data, msg_cls)


def convert_bag_to_omega_prime(
    bag_dir: Path,
    topic: str,
    output_dir: Path,
    child_frame: str,
    frame: str,
    map_path: Path | None = None,
    validate: bool = False,
) -> Path:
    projection_store: dict[int, dict[str, Any]] = {}

    def row_iter() -> Iterable[dict[str, Any]]:
        for msg in iter_object_list_messages(
            bag_dir,
            topic,
            child_frame,
            frame,
            projection_store=projection_store,
        ):
            yield from _olist_to_rows(msg)

    df = pl.DataFrame(row_iter())

    projections = _projection_store_to_projections(projection_store, frame)
    rec = omega_prime.Recording(df=df, projections=projections, validate=validate)

    if map_path and map_path.exists():
        rec.map = omega_prime.MapOdr.from_file(str(map_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{bag_dir.name}.omega-prime.mcap"
    rec.to_mcap(out_path)
    return out_path


def _discover_bags(data_dir: Path) -> list[Path]:
    bags = {path.parent for path in data_dir.rglob("metadata.yaml")}
    return sorted(bags)


def _parse_args() -> argparse.Namespace:
    env_validate = os.environ.get("OP_VALIDATE", "").lower() in {"1", "true", "yes"}

    parser = argparse.ArgumentParser(
        description="Convert ROS 2 ObjectList bags to omega-prime MCAP"
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("OP_DATA", "/data"),
        help="Directory containing rosbag2 folders",
    )
    parser.add_argument(
        "--topic", default=os.environ.get("OP_TOPIC"), help="ObjectList topic to export"
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OP_OUT", "/out"),
        help="Directory to write omega-prime MCAPs",
    )
    parser.add_argument(
        "--bag",
        action="append",
        default=[],
        help="Explicit bag directory to convert (repeatable)",
    )
    parser.add_argument(
        "--map",
        dest="map_path",
        default="/map/map.xodr",
        help="Optional OpenDRIVE map to embed",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=env_validate,
        help="Enable omega-prime schema validation",
    )
    parser.add_argument(
        "--child_frame",
        default="base_link",
        help="Source frame",
    )
    parser.add_argument(
        "--frame",
        default="utm_32N",
        help="Target frame",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.topic:
        raise SystemExit("--topic or OP_TOPIC env variable must be provided")

    bag_dirs = [Path(b).resolve() for b in args.bag]
    data_root = Path(args.data_dir).resolve()
    if data_root.exists():
        bag_dirs.extend(_discover_bags(data_root))

    unique = {}
    for bag in bag_dirs:
        if not bag.exists():
            raise FileNotFoundError(f"Bag path not found: {bag}")
        if not (bag / "metadata.yaml").exists():
            raise FileNotFoundError(f"metadata.yaml missing in bag directory: {bag}")
        unique[bag] = None

    bags = sorted(unique)
    if not bags:
        raise SystemExit("No rosbag2 directories with metadata.yaml found")

    out_dir = Path(args.output_dir).resolve()
    map_path = Path(args.map_path).resolve() if args.map_path else None

    for bag in bags:
        if map_path and map_path.exists():
            print(
                f"[object_list_to_omega_prime] Processing bag: {bag} with openDRIVE File: {map_path}"
            )
        else:
            print(
                f"[object_list_to_omega_prime] Processing bag: {bag} without openDRIVE File"
            )
        out_file = convert_bag_to_omega_prime(
            bag,
            args.topic,
            out_dir,
            args.child_frame,
            args.frame,
            map_path,
            args.validate,
        )
        print(f"[object_list_to_omega_prime] Wrote {out_file}")


if __name__ == "__main__":
    main()
