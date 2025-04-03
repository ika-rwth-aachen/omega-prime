from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

import omega_prime

app = typer.Typer(pretty_exceptions_show_locals=False)

app.registered_commands += omega_prime.converters.app.registered_commands


@app.command(help="Convert from ASAM OSI GroundTruth trace.")
def from_osi(
    input: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Path to ASAM OSI trace file (either `.osi` or `.mcap`)")
    ],
    output: Annotated[Path, typer.Argument(exists=False, dir_okay=False, help="Desired filename of omega file")],
    odr: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Path to ASAM OpenDRIVE xml to use as map")
    ] = None,
    validate: bool = True,
    skip_odr_parse: bool = False,
):
    r = omega_prime.Recording.from_file(input, xodr_path=odr, validate=validate, skip_odr_parse=skip_odr_parse)
    r.to_mcap(output)


@app.command(help="Convert from csv table according to schema")
def from_csv(
    input: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Path to csv according to omega moving object csv schema"),
    ],
    output: Annotated[Path, typer.Argument(exists=False, dir_okay=False, help="Desired filename of omega file")],
    odr: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Path to ASAM OpenDRIVE xml to use as map")
    ] = None,
    validate: bool = True,
    skip_odr_parse: bool = False,
):
    df = pd.read_csv(input)
    r = omega_prime.Recording(df, validate=validate)
    if odr is not None:
        r.map = omega_prime.asam_odr.MapOdr.from_file(odr, skip_parse=skip_odr_parse)
    r.to_mcap(output)


@app.command(help="Check an omega-prime file for specification conformance.")
def validate(
    input: Annotated[Path, typer.Argument(help="Path to omega file to validate", exists=True, dir_okay=False)],
):
    omega_prime.Recording.from_file(input)
    print(f"File {input} is valid.")


@app.command(help="Extracts the ASAM OpenDRIVE file from the omega-prime file.")
def to_odr(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Path to the omega-prime mcap file.")],
    output: Annotated[
        Path | None,
        typer.Argument(
            help="Where to store the ASAM OpenDRIVE file. If None or directory, stored filename will be used."
        ),
    ] = None,
):
    r = omega_prime.Recording.from_file(input, validate=False, skip_odr_parse=False)
    if isinstance(r.map, omega_prime.MapOdr):
        r.map.to_file(output)
    else:
        raise ValueError("The provided omega-prime file does not contain a map in ASAM OpenDRIVE format.")


if __name__ == "__main__":
    app()
