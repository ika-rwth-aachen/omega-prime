from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

import omega_format

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def from_osi(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Path to ASAM OSI trace file (either `.osi` or `.mcap`)")],
    output: Annotated[Path, typer.Argument(exists=False, dir_okay=False, help="Desired filename of omega file")],
    odr: Annotated[Path|None, typer.Option(exists=True, dir_okay=False, help="Path to ASAM OpenDRIVE xml to use as map")] = None,
    validate: bool = True, 
    skip_odr_parse: bool = False):
    r = omega_format.Recording.from_file(input, xodr_path=odr, validate=validate, skip_odr_parse=skip_odr_parse)
    r.to_mcap(output)
    
@app.command()
def from_csv(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Path to csv according to omega moving object csv schema")],
    output: Annotated[Path, typer.Argument(exists=False, dir_okay=False, help="Desired filename of omega file")],
    odr: Annotated[Path|None, typer.Option(exists=True, dir_okay=False, help="Path to ASAM OpenDRIVE xml to use as map")] = None,
    validate: bool = True, 
    skip_odr_parse: bool = False):
    df = pd.read_csv(input)
    r = omega_format.Recording(df, validate=validate)
    if odr is not None:
        r.map=omega_format.asam_odr.MapOdr.from_file(odr, skip_parse=skip_odr_parse)
    r.to_mcap(output)


@app.command()
def validate(
    input: Annotated[Path, typer.Argument(help="Path to omega file to validate", exists=True, dir_okay=False)]
):
    omega_format.Recording.from_file(input)
    print(f"File {input} is valid.")

if __name__ == "__main__":
    app()