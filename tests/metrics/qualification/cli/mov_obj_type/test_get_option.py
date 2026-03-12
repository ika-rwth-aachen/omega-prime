"""."""

import click
import typer.models

from omega_prime.metrics.qualification.cli.mov_obj_type import MovObjTypeCli


def test_get_option() -> None:
    typer_option = MovObjTypeCli.get_option()
    assert isinstance(typer_option, typer.models.OptionInfo)
    assert typer_option.help == "Select one or more expected moving object types."
    assert typer_option.default == "--moving-object-type"
    assert typer_option.param_decls == ("-t",)
    click_type = typer_option.click_type
    assert isinstance(click_type, click.Choice)
    assert click_type.choices == tuple(MovObjTypeCli.CHOICES)
