"""."""

from pathlib import Path

import click
import typer
from typing import Annotated

from omega_prime import Recording
from omega_prime.schemas import polars_schema

from .metric_dispatch import CliMetric, get_metric_manager
from .vehicle_type import VehicleType
from .mov_obj_type import MovObjTypeCli
from .vehicle_role import VehicleRole


H_FILE_NAME = "File name with Omega Prime data (.parquet, .osi or .mcap files)."
H_COLS = "Select one or more column names."

ctx_file_name = typer.Argument(dir_okay=False, file_okay=True, readable=True, help=H_FILE_NAME)
opt_fps = typer.Option("--fps", "-fps", help="Expected frequency.")
opt_columns = typer.Option("--columns", "-c", help=H_COLS, click_type=click.Choice(polars_schema.keys()))


class QualificationCli:
    def __init__(self) -> None:
        self.file_path: Path = Path("")

    def entry_point(
        self,
        file_path: Annotated[Path, ctx_file_name],
        cli_metrics: Annotated[list[CliMetric], CliMetric.get_option()] = CliMetric.get_default(),
        types: Annotated[list[str], MovObjTypeCli.get_option()] = MovObjTypeCli.get_default(),
        subtypes: Annotated[list[str], VehicleType.get_option()] = VehicleType.get_default(),
        roles: Annotated[list[str], VehicleRole.get_option()] = VehicleRole.get_default(),
        fps: Annotated[float, opt_fps] = 30.0,
        columns: Annotated[list[str], opt_columns] = tuple(polars_schema.keys()),
    ) -> None:
        self.file_path = file_path
        manager = get_metric_manager(cli_metrics)
        recording = Recording.from_file(file_path)
        expected_types = MovObjTypeCli.to_osi_list(types)
        expected_subtypes = VehicleType.to_osi_list(subtypes)
        expected_roles = VehicleRole.to_osi_list(roles)
        _df, results_dct = manager.compute(
            recording,
            expected_types=expected_types,
            expected_subtypes=expected_subtypes,
            expected_roles=expected_roles,
            expected_frequency=fps,
            columns=columns,
        )
        for key, frame in results_dct.items():
            print(key, "\n", frame)


qualification_cli = QualificationCli()
qualification_typer = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)
command = qualification_typer.command("qualify", help="Qualify Omega Prime dataframe.")
command(qualification_cli.entry_point)
