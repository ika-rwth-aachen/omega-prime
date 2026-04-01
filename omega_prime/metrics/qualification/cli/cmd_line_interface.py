"""."""

from datetime import datetime
from pathlib import Path

import click
import typer
from typing import Annotated

from omega_prime import Recording
from omega_prime.schemas import polars_schema

from .metric_dispatch import CliMetric, get_metric_manager

from .mov_obj_type import MovObjTypeCli
from .vehicle_role import VehicleRoleCli
from .vehicle_type import VehicleTypeCli

from .class_completeness import ClassCompletenessCli
from .non_default_attr_accuracy import NonDefaultAttributesAccuracyCli
from .object_type_coverage import ObjectTypeCoverageCli
from .target_area_coverage import TargetAreaCoverageCli
from .temporal_completeness import TemporalCompletenessCli
from .temporal_coverage import TemporalCoverageCli


H_FILE_NAME = "File name with Omega Prime data (.parquet, .osi or .mcap files)."
H_COLS = "Select one or more column names."
H_THRESHOLD = "Optional threshold for coverage metrics."

ctx_file_name = typer.Argument(dir_okay=False, file_okay=True, readable=True, help=H_FILE_NAME)
opt_fps = typer.Option("--fps", "-fps", help="Expected frequency.")
opt_columns = typer.Option("--columns", "-c", help=H_COLS, click_type=click.Choice(polars_schema.keys()))
opt_threshold = typer.Option("--threshold", help=H_THRESHOLD)


class QualificationCli:
    def __init__(self) -> None:
        self.file_path: Path = Path("")

    @classmethod
    def _get_metric_kwargs(
        cls,
        recording: object,
        cli_metrics: list[CliMetric],
        types: list[str],
        subtypes: list[str],
        roles: list[str],
        fps: float,
        columns: list[str],
        expected_start: datetime | None,
        expected_end: datetime | None,
        expected_area_coords: list[object],
        expected_area_source_crs: str | None,
        threshold: float | None,
    ) -> dict[str, object]:
        metric_kwargs: dict[str, object] = {}
        selected_metrics = set(cli_metrics)

        if threshold is not None:
            metric_kwargs["threshold"] = threshold

        if CliMetric.CLASS_COMPLETENESS in selected_metrics:
            metric_kwargs |= ClassCompletenessCli.build_kwargs(types, subtypes, roles)

        if CliMetric.OBJECT_TYPE_COVERAGE in selected_metrics:
            metric_kwargs |= ObjectTypeCoverageCli.build_kwargs(types)

        if CliMetric.TEMPORAL_COMPLETENESS in selected_metrics:
            metric_kwargs |= TemporalCompletenessCli.build_kwargs(fps)

        if CliMetric.NON_DEFAULT_ATTRIBUTES_ACCURACY in selected_metrics:
            metric_kwargs |= NonDefaultAttributesAccuracyCli.build_kwargs(columns)

        if CliMetric.TEMPORAL_COVERAGE in selected_metrics:
            metric_kwargs |= TemporalCoverageCli.build_kwargs(expected_start, expected_end)

        if CliMetric.TARGET_AREA_COVERAGE in selected_metrics:
            metric_kwargs |= TargetAreaCoverageCli.build_kwargs(
                recording=recording,
                expected_area_coords=expected_area_coords,
                expected_area_source_crs=expected_area_source_crs,
            )

        return metric_kwargs

    def entry_point(
        self,
        file_path: Annotated[Path, ctx_file_name],
        cli_metrics: Annotated[list[CliMetric], CliMetric.get_option()] = CliMetric.get_default(),
        types: Annotated[list[str], MovObjTypeCli.get_option()] = MovObjTypeCli.get_default(),
        subtypes: Annotated[list[str], VehicleTypeCli.get_option()] = VehicleTypeCli.get_default(),
        roles: Annotated[list[str], VehicleRoleCli.get_option()] = VehicleRoleCli.get_default(),
        fps: Annotated[float, opt_fps] = 30.0,
        columns: Annotated[list[str], opt_columns] = tuple(polars_schema.keys()),
        expected_start: Annotated[datetime | None, TemporalCoverageCli.get_expected_start_option()] = None,
        expected_end: Annotated[datetime | None, TemporalCoverageCli.get_expected_end_option()] = None,
        expected_area_coords: Annotated[list[object], TargetAreaCoverageCli.get_expected_area_coord_option()] = tuple(),
        expected_area_source_crs: Annotated[
            str | None,
            TargetAreaCoverageCli.get_expected_area_source_crs_option(),
        ] = None,
        threshold: Annotated[float | None, opt_threshold] = None,
    ) -> None:
        self.file_path = file_path
        recording = Recording.from_file(file_path)
        metric_kwargs = self._get_metric_kwargs(
            recording=recording,
            cli_metrics=cli_metrics,
            types=types,
            subtypes=subtypes,
            roles=roles,
            fps=fps,
            columns=list(columns),
            expected_start=expected_start,
            expected_end=expected_end,
            expected_area_coords=list(expected_area_coords),
            expected_area_source_crs=expected_area_source_crs,
            threshold=threshold,
        )
        manager = get_metric_manager(cli_metrics)
        _df, results_dct = manager.compute(
            recording,
            **metric_kwargs,
        )
        for key, frame in results_dct.items():
            print(key, "\n", frame)


qualification_cli = QualificationCli()
qualification_typer = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)
command = qualification_typer.command("qualify", help="Qualify Omega Prime dataframe.")
command(qualification_cli.entry_point)
