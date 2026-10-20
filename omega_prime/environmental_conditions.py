"OSI `EnvironmentalConditions` of a recording: storage in the Parquet metadata and tabular view."

import copy
import json
import typing

import betterosi
import numpy as np
import polars as pl

from .schemas import polars_schema

# `type` of the `betterosi.ExternalReference` in `EnvironmentalConditions.source_reference` that carries a measured
# precipitation value. The value is stored as a string in the first `identifier`.
PRECIPITATION_INTENSITY_REFERENCE_TYPE = "omega_prime.precipitation_intensity_mm_per_h"
PRECIPITATION_AMOUNT_REFERENCE_TYPE = "omega_prime.precipitation_amount_mm"

SCHEMA = {
    "total_nanos": polars_schema["total_nanos"],
    "frame": pl.UInt32,
    "precipitation": pl.Int64,
    "precipitation_intensity": pl.Float64,
    "precipitation_amount": pl.Float64,
    "fog": pl.Int64,
    "ambient_illumination": pl.Int64,
    "temperature": pl.Float64,
    "atmospheric_pressure": pl.Float64,
    "relative_humidity": pl.Float64,
    "unix_timestamp": pl.Int64,
    "seconds_since_midnight": pl.Int64,
    "fractional_cloud_cover": pl.Int64,
    "wind_origin_direction": pl.Float64,
    "wind_speed": pl.Float64,
    "sun_azimuth": pl.Float64,
    "sun_elevation": pl.Float64,
    "sun_intensity": pl.Float64,
}


def encode(
    environmental_conditions: dict[int, betterosi.EnvironmentalConditions],
    total_nanos: typing.Iterable[int],
) -> bytes:
    """
    Encode environmental conditions into a JSON string and then to bytes.
    Consecutive timestamps of `total_nanos` with equal conditions are stored as one run.
    `version` is the version of this encoding.
    """
    runs: list[dict[str, typing.Any]] = []
    current = None
    for ts in sorted(int(n) for n in total_nanos):
        conditions = environmental_conditions.get(ts)
        if conditions is None:
            current = None
        elif current is not None and current[0] == conditions:
            current[1]["end"] = ts
        else:
            run = {"start": ts, "end": ts, "conditions": conditions.to_dict()}
            runs.append(run)
            current = (conditions, run)
    if not runs:
        return b""
    return json.dumps({"version": 1, "runs": runs}).encode()


def decode(
    raw: bytes | str | None,
    total_nanos: typing.Iterable[int],
) -> dict[int, betterosi.EnvironmentalConditions]:
    """
    Decode environmental conditions from bytes or string to a dictionary keyed by the given `total_nanos`.
    Each timestamp gets its own `betterosi.EnvironmentalConditions`.
    """
    if raw in (None, b"", ""):
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode()
    payload = json.loads(raw)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported version {payload.get('version')!r} of the environmental conditions.")
    timestamps = np.array(sorted(int(n) for n in total_nanos), dtype=np.int64)
    result: dict[int, betterosi.EnvironmentalConditions] = {}
    for run in payload.get("runs", []):
        conditions = betterosi.EnvironmentalConditions().from_dict(run["conditions"])
        lo = np.searchsorted(timestamps, int(run["start"]), side="left")
        hi = np.searchsorted(timestamps, int(run["end"]), side="right")
        for ts in timestamps[lo:hi]:
            result[int(ts)] = copy.deepcopy(conditions)
    return result


def source_reference_value(conditions: betterosi.EnvironmentalConditions, reference_type: str) -> float | None:
    "Value of the first `source_reference` of the given type, None if there is none or it is not a number."
    for reference in conditions.source_reference:
        if reference.type == reference_type and reference.identifier:
            try:
                return float(reference.identifier[0])
            except ValueError:
                return None
    return None


def to_df(
    environmental_conditions: dict[int, betterosi.EnvironmentalConditions], nanos2frame: dict[int, int]
) -> pl.DataFrame:
    "One row per timestamp of `environmental_conditions`, see `Recording.environmental_conditions_df`."
    rows = []
    for nanos, ec in sorted(environmental_conditions.items()):
        rows.append(
            dict(
                total_nanos=nanos,
                frame=nanos2frame.get(nanos),
                precipitation=int(ec.precipitation),
                precipitation_intensity=source_reference_value(ec, PRECIPITATION_INTENSITY_REFERENCE_TYPE),
                precipitation_amount=source_reference_value(ec, PRECIPITATION_AMOUNT_REFERENCE_TYPE),
                fog=int(ec.fog),
                ambient_illumination=int(ec.ambient_illumination),
                temperature=ec.temperature or None,
                atmospheric_pressure=ec.atmospheric_pressure or None,
                relative_humidity=ec.relative_humidity or None,
                unix_timestamp=ec.unix_timestamp or None,
                seconds_since_midnight=None if ec.time_of_day is None else ec.time_of_day.seconds_since_midnight,
                fractional_cloud_cover=None if ec.clouds is None else int(ec.clouds.fractional_cloud_cover),
                wind_origin_direction=None if ec.wind is None else ec.wind.origin_direction,
                wind_speed=None if ec.wind is None else ec.wind.speed,
                sun_azimuth=None if ec.sun is None else ec.sun.azimuth,
                sun_elevation=None if ec.sun is None else ec.sun.elevation,
                sun_intensity=None if ec.sun is None else ec.sun.intensity,
            )
        )
    return pl.DataFrame(rows, schema=SCHEMA)
