"""."""

import click
import typer.models

from omega_prime.metrics.qualification.cli.target_area_coverage import (
    H_EXPECTED_AREA_COORD,
    H_EXPECTED_AREA_SOURCE_CRS,
    TargetAreaCoverageCli,
)


def test_get_expected_area_coord_option() -> None:
    typer_option = TargetAreaCoverageCli.get_expected_area_coord_option()
    assert isinstance(typer_option, typer.models.OptionInfo)
    assert typer_option.help == H_EXPECTED_AREA_COORD
    assert typer_option.default == "--expected-area-coord"
    assert typer_option.param_decls == ()
    click_type = typer_option.click_type
    assert isinstance(click_type, click.Tuple)
    assert click_type.types == [click.FLOAT, click.FLOAT]


def test_get_expected_area_source_crs_option() -> None:
    typer_option = TargetAreaCoverageCli.get_expected_area_source_crs_option()
    assert isinstance(typer_option, typer.models.OptionInfo)
    assert typer_option.help == H_EXPECTED_AREA_SOURCE_CRS
    assert typer_option.default == "--expected-area-source-crs"
    assert typer_option.param_decls == ()
    assert typer_option.click_type is None
