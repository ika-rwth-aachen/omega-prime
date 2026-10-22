"Recordings and environmental conditions shared by the tests of `Recording`."

import json
from pathlib import Path

import betterosi

import omega_prime

p = Path(__file__).parent.parent / "example_files/"
with open(p / "mapping.json") as f:
    mapping = json.load(f)

EC = betterosi.EnvironmentalConditions
Precipitation = betterosi.EnvironmentalConditions.Precipitation
Fog = betterosi.EnvironmentalConditions.Fog

MODERATE = EC(precipitation=Precipitation.MODERATE, fog=Fog.GOOD_VISIBILITY, temperature=283.15)
HEAVY = EC(
    precipitation=Precipitation.HEAVY,
    fog=Fog.MIST,
    wind=betterosi.EnvironmentalConditions.Wind(origin_direction=1.5, speed=0.0),
)


def load(i: int = 3, **kwargs) -> omega_prime.Recording:
    return omega_prime.Recording.from_file(p / mapping[i][0], p / mapping[i][1], validate=False, **kwargs)


def set_time_varying(rec: omega_prime.Recording) -> list[int]:
    "Set MODERATE on the first half and HEAVY on the second half of the recording. Returns the sorted timestamps."
    nanos = sorted(rec.nanos2frame)
    rec.set_environmental_conditions(MODERATE)
    rec.set_environmental_conditions(HEAVY, nanos=nanos[len(nanos) // 2 :])
    return nanos


class ExplicitInitRecording(omega_prime.Recording):
    "Subclass with an `__init__` that does not know the argument `environmental_conditions`."

    def __init__(
        self, df, map=None, projections=None, host_vehicle_idx=None, validate=False, traffic_light_states=None
    ):
        super().__init__(
            df,
            map=map,
            projections=projections,
            host_vehicle_idx=host_vehicle_idx,
            validate=validate,
            traffic_light_states=traffic_light_states,
        )
