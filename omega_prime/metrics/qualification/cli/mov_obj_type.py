"""."""

import click
import typer
from betterosi import MovingObject

from .common import to_osi_str

H_MOV_OBJ_TYPE = "Select one or more expected moving object types."


class MovObjTypeCli:
    CHOICES = "animal", "other", "pedestrian", "unknown", "vehicle"

    @staticmethod
    def to_osi_list(obj_types: list[str]) -> list[MovingObject.Type]:
        return [MovingObject.Type[to_osi_str(t)] for t in obj_types]

    @staticmethod
    def get_default() -> tuple[str, str]:
        return "pedestrian", "vehicle"

    @classmethod
    def get_option(cls) -> typer.models.OptionInfo:
        return typer.Option(
            "--moving-object-type",
            "-t",
            help=H_MOV_OBJ_TYPE,
            click_type=click.Choice(cls.CHOICES),
        )
