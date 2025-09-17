__pdoc__ = {}
__pdoc__["converters"] = False
""" .. include:: ./../README.md """
import importlib.util
import warnings
from . import converters, metrics
from .map_odr import MapOdr
from .locator import LaneRelation, Locator
from .map import Lane, LaneBoundary, Map, MapOsi
from .recording import MovingObject, Recording
from importlib.metadata import version

if importlib.util.find_spec("lanelet2") is not None:
    from .map_lanelet import MapLanelet
else:

    class MapLanelet:
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "lanelet2 is not installed. To use MapLanelet you need to install `lanelet2` with `pip install lanelet2` (only available on linux). If you are using windows and python <= 3.12 you can try to install `lanelet2x` instead."
            )

        def __getattr__(self, name):
            raise RuntimeError(
                f"Cannot use attribute '{name}', since lanelet2 is not installed. To use MapLanelet you need to install `lanelet2` with `pip install lanelet2` (only available on linux). If you are using windows and python <= 3.12 you can try to install `lanelet2x` instead."
            )


__version__ = version("omega_prime")

__all__ = [
    "Recording",
    "MovingObject",
    "MapOsi",
    "Map",
    "Lane",
    "LaneBoundary",
    "MapLanelet",
    "MapOdr",
    "Locator",
    "LaneRelation",
    "converters",
    "metrics",
]
