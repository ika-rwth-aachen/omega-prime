"""
Standalone converter: read perception_msgs/ObjectList messages from ROS 2 bags
and emit omega-prime MCAP files without relying on ros2_unbag.

The CLI can process specific bag directories or scan a data root for rosbag2
folders (identified via metadata.yaml).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import yaml
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from rclpy.time import Time

import numpy as np
import polars as pl
import omega_prime
import betterosi

import perception_msgs_utils as pmu

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

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


def _quat_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _object_to_row(obj) -> Dict[str, Any]:
    total_nanos = Time.from_msg(obj.state.header.stamp).nanoseconds

    pos = pmu.get_position(obj)
    width = pmu.get_width(obj)
    length = pmu.get_length(obj)
    height = pmu.get_height(obj)

    q = pmu.get_orientation(obj)
    roll, pitch, yaw = _quat_to_euler(float(q.x), float(q.y), float(q.z), float(q.w))

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


def _olist_to_rows(msg) -> List[Dict[str, Any]]:
    return [_object_to_row(obj) for obj in msg.objects]


def _load_metadata(bag_dir: Path) -> Dict[str, Any]:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found in {bag_dir}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _storage_id(meta: Dict[str, Any]) -> str:
    return meta["rosbag2_bagfile_information"]["storage_identifier"]


def _topic_type_map(reader: SequentialReader) -> Dict[str, str]:
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


def iter_object_list_messages(bag_dir: Path, topic: str) -> Iterator[Any]:
    meta = _load_metadata(bag_dir)
    storage_id = _storage_id(meta)

    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(bag_dir), storage_id=storage_id)
    converter_options = ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    type_map = _topic_type_map(reader)
    if topic not in type_map:
        available = ", ".join(sorted(type_map))
        raise RuntimeError(f"Topic {topic} not found. Available topics: {available}")

    msg_cls = get_message(type_map[topic])

    while reader.has_next():
        topic_name, data, _ = reader.read_next()
        if topic_name != topic:
            continue
        yield deserialize_message(data, msg_cls)


def convert_bag_to_omega_prime(
    bag_dir: Path,
    topic: str,
    output_dir: Path,
    map_path: Path | None = None,
    validate: bool = False,
) -> Path:
    def row_iter() -> Iterable[Dict[str, Any]]:
        for msg in iter_object_list_messages(bag_dir, topic):
            yield from _olist_to_rows(msg)

    df = pl.DataFrame(row_iter())
    rec = omega_prime.Recording(df=df, validate=validate)

    if map_path and map_path.exists():
        rec.map = omega_prime.MapOdr.from_file(str(map_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{bag_dir.name}.omega-prime.mcap"
    rec.to_mcap(out_path)
    return out_path


def _discover_bags(data_dir: Path) -> List[Path]:
    bags = {path.parent for path in data_dir.rglob("metadata.yaml")}
    return sorted(bags)


def _parse_args() -> argparse.Namespace:
    env_validate = os.environ.get("OP_VALIDATE", "").lower() in {"1", "true", "yes"}

    parser = argparse.ArgumentParser(description="Convert ROS 2 ObjectList bags to omega-prime MCAP")
    parser.add_argument("--data-dir", default=os.environ.get("OP_DATA", "/data"), help="Directory containing rosbag2 folders")
    parser.add_argument("--topic", default=os.environ.get("OP_TOPIC"), help="ObjectList topic to export")
    parser.add_argument("--output-dir", default=os.environ.get("OP_OUT", "/out"), help="Directory to write omega-prime MCAPs")
    parser.add_argument("--bag", action="append", default=[], help="Explicit bag directory to convert (repeatable)")
    parser.add_argument("--map", dest="map_path", default=os.environ.get("OP_ODR"), help="Optional OpenDRIVE map to embed")
    parser.add_argument("--validate", action="store_true", default=env_validate, help="Enable omega-prime schema validation")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.topic:
        raise SystemExit("--topic or OP_TOPIC env variable must be provided")

    bag_dirs = [Path(b).resolve() for b in args.bag]
    data_root = Path(args.data_dir).resolve()
    if data_root.exists():
        bag_dirs.extend(_discover_bags(data_root))

    unique: Dict[Path, None] = {}
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
        out_file = convert_bag_to_omega_prime(bag, args.topic, out_dir, map_path, args.validate)
        print(f"[object_list_to_omega_prime] Wrote {out_file}")


if __name__ == "__main__":
    main()
