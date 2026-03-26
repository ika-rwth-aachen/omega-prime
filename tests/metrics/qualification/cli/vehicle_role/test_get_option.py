"""."""

import click
import typer.models

from omega_prime.metrics.qualification.cli.vehicle_role import VehicleRole


def test_get_option() -> None:
    typer_option = VehicleRole.get_option()
    assert isinstance(typer_option, typer.models.OptionInfo)
    assert typer_option.help == "Select one or more expected vehicle roles."
    assert typer_option.default == "--vehicle-role"
    assert typer_option.param_decls == ("-r",)
    click_type = typer_option.click_type
    assert isinstance(click_type, click.Choice)
    assert click_type.choices == VehicleRole.CHOICES
