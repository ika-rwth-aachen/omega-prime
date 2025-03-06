try:
    from .lxd import convert_lxd
    lxd_installed = True
except ImportError:
    lxd_installed = False
    
from pathlib import Path
from typing import Annotated

import typer

__all__ = ['app'] + (['convert_lxd'] if lxd_installed else [])

app = typer.Typer(pretty_exceptions_show_locals=False)

if lxd_installed:
    @app.command('from-lxd', help='Convert datasets from LevelXData to omega-prime.')
    def convert_lxd_cli(
        dataset_path: Annotated[Path, typer.Argument(exists=True, dir_okay=True, file_okay=False, readable=True, help='Root of the LevelXData dataset')],
        output_path: Annotated[Path, typer.Argument(file_okay=False, writable=True, help='In which folder to write the created omega-prime files')]
    ):
        Path(output_path).mkdir(exist_ok=True)
        convert_lxd(dataset_dir=dataset_path, output_path=output_path)
else:
    @app.command('from-lxd', help='Install `omega-prime[lxd]` to convert LevelXData datasets with this command.')
    def mock(
        dataset_path: Annotated[Path, typer.Argument(exists=True, dir_okay=True, file_okay=False, readable=True, help='Root of the LevelXData dataset')],
        output_path: Annotated[Path, typer.Argument(file_okay=False, writable=True, help='In which folder to write the created omega-prime files')]
    ):
        raise RuntimeError('You need to have `lxd-io` installed to run the conversion of LevelXData. Run `pip install omega-prime[lxd]`.')