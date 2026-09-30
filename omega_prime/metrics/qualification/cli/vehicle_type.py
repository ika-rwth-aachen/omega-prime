"""."""

import click
import typer.models
import betterosi

from .common import to_osi_str

OsiCls = betterosi.MovingObject.VehicleClassification.Type


class VehicleTypeCli:
    # fmt: off
    CHOICES = (
        'aircraft', 'bicycle', 'bus', 'car', 'compact-car', 'delivery-van', 'heavy-truck', 'land-vehicle',
        'luxury-car', 'medium-car', 'micromobility-device', 'motorbike', 'motorcycle', 'other',
        'semitractor', 'semitrailer', 'small-car', 'standup-scooter', 'trailer', 'train', 'tram',
        'unknown', 'van', 'watercraft', 'wheelchair', 'work-machine'
    )
    # fmt: on

    @staticmethod
    def to_osi_list(veh_cls: list[str]) -> list[OsiCls]:
        return [OsiCls[to_osi_str(c)] for c in veh_cls]

    @classmethod
    def get_option(cls) -> typer.models.OptionInfo:
        return typer.Option(
            "--vehicle-class",
            "-v",
            help="Select one or more expected vehicle classes.",
            click_type=click.Choice(cls.CHOICES),
        )

    @staticmethod
    def get_default() -> tuple[str, ...]:
        return tuple()
