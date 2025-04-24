from .lxd import LxdConverter

from pathlib import Path
from typing import Annotated

import typer

__all__ = ["app"]

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("from-lxd", help="Convert datasets from LevelXData to omega-prime.")
def convert_lxd_cli(
    dataset_path: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=True, file_okay=False, readable=True, help="Root of the LevelXData dataset"
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(file_okay=False, writable=True, help="In which folder to write the created omega-prime files"),
    ],
    n_workers: Annotated[int, typer.Option(help="Set to -1 for n_cpus-1 workers.")] = 1,
    save_as_parquet: Annotated[
        bool,
        typer.Option(
            help="If activated, omega-prime recordings will be stored as parquet files instead of mcap (use for large recordings). Will loose information in OSI that are not mandatory in omega-prime."
        ),
    ] = False,
):
    Path(output_path).mkdir(exist_ok=True)
    LxdConverter(dataset_path=dataset_path, out_path=output_path, n_workers=n_workers).convert(
        save_as_parquet=save_as_parquet
    )
