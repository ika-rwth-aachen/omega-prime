from .lxd import convert_lxd

from pathlib import Path
from typing import Annotated

import typer

__all__ = ['app', 'convert_lxd']

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command('from-lxd', help='Convert datasets from LevelXData to omega-prime.')
def convert_lxd_cli(
    dataset_path: Annotated[Path, typer.Argument(exists=True, dir_okay=True, file_okay=False, readable=True, help='Root of the LevelXData dataset')],
    output_path: Annotated[Path, typer.Argument(file_okay=False, writable=True, help='In which folder to write the created omega-prime files')],
    n_workers: Annotated[int, typer.Option(help='Set to -1 for n_cpus-1 workers.')] = 1
):
    Path(output_path).mkdir(exist_ok=True)
    convert_lxd(dataset_dir=dataset_path, outpath=output_path, n_workers=n_workers)