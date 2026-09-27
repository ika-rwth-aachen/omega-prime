"""."""

import click
import typer.models
import betterosi

OsiRole = betterosi.MovingObject.VehicleClassification.Role

from .common import to_osi_str


class VehicleRoleCli:
    # fmt: off
    CHOICES = (
        'ambulance', 'civil', 'fire', 'garbage-collection', 'military', 'other', 'police', 'public-transport',
        'road-assistance', 'road-construction', 'unknown'
    )
    # fmt: on

    @staticmethod
    def to_osi_list(roles: list[str]) -> list[OsiRole]:
        return [OsiRole[to_osi_str(role)] for role in roles]

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
