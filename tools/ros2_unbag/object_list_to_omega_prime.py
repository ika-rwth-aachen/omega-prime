"""
ros2_unbag export routine: Convert perception_msgs/msg/ObjectList streams into
an omega-prime MCAP using a Polars DataFrame and omega-prime's Recording API.

Usage with ros2_unbag (example):
  ros2 unbag my_bag.mcap \
    --use-routine /path/to/tools/ros2_unbag/object_list_to_omega_prime.py \
    --export /perception/object_list:omega-prime/MCAP \
    --output out_dir

Assumptions:
- Bags contain ObjectList with per-object state timestamps.
- No OpenDRIVE map is present; the exported MCAP will not include a map.

Notes:
- The routine declares SINGLE_FILE mode so ros2_unbag passes an iterator
  of messages; we also handle the case where messages arrive one-by-one
  (fallback accumulation) if ros2_unbag behaves differently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any

# ros2_unbag runtime decorator/types (available in ros2_unbag environment)
try:
    from ros2_unbag.core.routines.base import ExportRoutine, ExportMode, ExportMetadata
except Exception:  # Allow local import (tests, linters) without ros2_unbag installed
    def ExportRoutine(*args, **kwargs):  # type: ignore
        def _wrap(func):
            return func
        return _wrap

    class ExportMode:  # type: ignore
        MULTI_FILE = object()
        SINGLE_FILE = object()

    class ExportMetadata:  # type: ignore
        pass

# ROS message helpers
from rclpy.time import Time
from tf_transformations import euler_from_quaternion

# perception utils (from perception_interfaces repo)
import perception_msgs_utils as pmu

# core libs
import os
import polars as pl
import omega_prime
import betterosi


# --- Classification mapping -------------------------------------------------

_VCT = betterosi.MovingObjectVehicleClassificationType
_ROLE = betterosi.MovingObjectVehicleClassificationRole
_MOT = betterosi.MovingObjectType


def _class_to_osi(obj) -> tuple[int, int, int]:
    """Map perception_msgs ObjectClassification to OSI type/role/subtype.

    Returns (type, role, subtype) as integers per omega-prime schema.
    If no classification exists, falls back to UNKNOWN/OTHER conservative defaults.
    """
    # Default fallbacks
    mot = int(_MOT.TYPE_OTHER)
    role = -1
    subtype = -1

    # classification with highest probability (if any)
    if obj.state.classifications:
        c = pmu.get_class_with_highest_probability(obj)
        ct = int(c.type)
    else:
        ct = 0  # UNKNOWN

    # perception_msgs classifications:
    # UNKNOWN=100 or UNCLASSIFIED=0, PEDESTRIAN=1, BICYCLE=2, MOTORBIKE=3,
    # CAR=4, TRUCK=5, VAN=6, BUS=7, ANIMAL=8, ROAD_OBSTACLE=9, TRAIN=10, TRAILER=11, ...

    vehicle_map = {
        4: _VCT.TYPE_CAR,              # CAR
        5: _VCT.TYPE_HEAVY_TRUCK,      # TRUCK
        6: _VCT.TYPE_DELIVERY_VAN,     # VAN
        7: _VCT.TYPE_BUS,              # BUS
        10: _VCT.TYPE_TRAIN,           # TRAIN
        3: _VCT.TYPE_MOTORBIKE,        # MOTORBIKE
        2: _VCT.TYPE_BICYCLE,          # BICYCLE
        11: _VCT.TYPE_TRAILER,         # TRAILER
        # Union types (50+) map to OTHER vehicle subtype to keep schema valid
        50: _VCT.TYPE_OTHER,           # CAR_UNION
        51: _VCT.TYPE_OTHER,           # TRUCK_UNION
        52: _VCT.TYPE_OTHER,           # BIKE_UNION
    }

    if ct == 1:  # PEDESTRIAN
        mot = int(_MOT.TYPE_PEDESTRIAN)
        role = -1
        subtype = -1
    elif ct == 8:  # ANIMAL
        mot = int(_MOT.TYPE_ANIMAL)
        role = -1
        subtype = -1
    elif ct in vehicle_map:
        mot = int(_MOT.TYPE_VEHICLE)
        role = int(_ROLE.ROLE_CIVIL)
        subtype = int(vehicle_map[ct])
    elif ct in (0, 9, 100):  # UNCLASSIFIED/ROAD_OBSTACLE/UNKNOWN
        mot = int(_MOT.TYPE_OTHER)
        role = -1
        subtype = -1
    else:
        # Fallback: treat as OTHER
        mot = int(_MOT.TYPE_OTHER)
        role = -1
        subtype = -1

    return mot, role, subtype


# --- Row extraction ---------------------------------------------------------

def _object_to_row(obj) -> Dict[str, Any]:
    # Timestamp (ns) from state header
    total_nanos = Time.from_msg(obj.state.header.stamp).nanoseconds

    # Pose/size
    pos = pmu.get_position(obj)
    width = pmu.get_width(obj)
    length = pmu.get_length(obj)
    height = pmu.get_height(obj)

    # Orientation yaw (rad)
    q = pmu.get_orientation(obj)
    roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    # Vel/acc in x/y (global frame)
    vel_x = pmu.get_vel_x(obj)
    vel_y = pmu.get_vel_y(obj)
    acc_x = pmu.get_acc_x(obj)
    acc_y = pmu.get_acc_y(obj)

    # Classification mapping
    mot, role, subtype = _class_to_osi(obj)

    return {
        "total_nanos": int(total_nanos),
        "idx": int(obj.id),
        "x": float(pos.x),
        "y": float(pos.y),
        "z": float(pos.z) if hasattr(pos, "z") else 0.0,
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


# --- Export routine ---------------------------------------------------------

@ExportRoutine("perception_msgs/msg/ObjectList", ["omega-prime/MCAP"], mode=ExportMode.SINGLE_FILE)
def export_object_list_to_omega_prime(msg_or_iter, path: Path, fmt: str, metadata: ExportMetadata):
    """Export routine invoked by ros2_unbag.

    - In SINGLE_FILE mode, ros2_unbag typically provides an iterable of messages.
    - If a single message is passed (non-iterable), we accumulate rows and write when possible.
    """
    out_path = Path(f"{path}.mcap")

    # Detect an iterable of messages vs a single message instance
    is_single = hasattr(msg_or_iter, "objects")

    if is_single:
        # Fallback: accumulate one message at a time, write immediately (best-effort)
        rows = _olist_to_rows(msg_or_iter)
        df = pl.DataFrame(rows)
        rec = omega_prime.Recording(df=df, validate=False)
        rec.to_mcap(out_path)
        return

    # Iterable: aggregate all messages into a single DataFrame stream
    def row_iter():
        for msg in msg_or_iter:  # type: ignore[union-attr]
            for row in _olist_to_rows(msg):
                yield row

    df = pl.DataFrame(row_iter())

    # Use omega-prime to write MCAP; optionally attach OpenDRIVE map via env OP_ODR
    rec = omega_prime.Recording(df=df, validate=False)
    odr_path = os.getenv("OP_ODR")
    if odr_path and os.path.exists(odr_path):
        try:
            rec.map = omega_prime.MapOdr.from_file(odr_path)
        except Exception:
            pass
    rec.to_mcap(out_path)
