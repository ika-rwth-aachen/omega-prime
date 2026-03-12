"""."""

import click
import typer.models
from betterosi import MovingObjectVehicleClassificationRole as OsiRole

from .common import to_osi_str


class VehicleRole:
    # fmt: off
    CHOICES = (
        'ambulance', 'civil', 'fire', 'garbage-collection', 'military', 'other', 'police', 'public-transport',
        'road-assistance', 'road-construction', 'unknown'
    )
    # fmt: on

    @staticmethod
    def to_osi_list(roles: list[str]) -> list[OsiRole]:
        return [OsiRole.from_string(to_osi_str(role)) for role in roles]

    @classmethod
    def get_option(cls) -> typer.models.OptionInfo:
        return typer.Option(
            "--vehicle-role",
            "-r",
            help="Select one or more expected vehicle roles.",
            click_type=click.Choice(cls.CHOICES),
        )

    @staticmethod
    def get_default() -> tuple[str, ...]:
        return tuple()
