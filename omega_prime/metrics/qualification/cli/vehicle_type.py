"""."""

import click
import typer.models
from betterosi import MovingObjectVehicleClassificationType as OsiCls

from .common import to_osi_str


class VehicleType:
    # fmt: off
    CHOICES = (
        'bicycle', 'bus', 'car', 'compact-car', 'delivery-van', 'heavy-truck', 'luxury-car', 'medium-car',
        'motorbike', 'other', 'semitractor', 'semitrailer', 'small-car', 'standup-scooter', 'trailer', 'train',
        'tram', 'unknown', 'wheelchair'
    )
    # fmt: on

    @staticmethod
    def to_osi_list(veh_cls: list[str]) -> list[OsiCls]:
        return [OsiCls.from_string(to_osi_str(c)) for c in veh_cls]

    @classmethod
    def get_option(cls) -> typer.models.OptionInfo:
        return typer.Option(
            "--vehicle-class",
            "-v",
            help="Select one or more vehicle classes.",
            click_type=click.Choice(cls.CHOICES),
        )

    @staticmethod
    def get_default() -> tuple[str, ...]:
        return tuple()
