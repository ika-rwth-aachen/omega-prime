"""Command line interface for omega-prime conversions and utilities."""
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from pathlib import Path
from mcap_protobuf.decoder import DecoderFactory
from mcap.reader import make_reader
import omega_prime
from omega_prime.converters import load_converters_into_cli
import altair as alt

app = typer.Typer(pretty_exceptions_show_locals=False)

app.registered_commands += omega_prime.converters.app.registered_commands


@app.command(help="Convert from ASAM OSI GroundTruth trace.")
def from_osi(
    input: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Path to ASAM OSI trace file (either `.osi` or `.mcap`)")
    ],
    output: Annotated[Path, typer.Argument(exists=False, dir_okay=False, help="Desired filename of omega file")],
    map_path: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Path to ASAM OpenDRIVE xml to use as map")
    ] = None,
    validate: bool = True,
):
    r = omega_prime.Recording.from_file(input, map_path=map_path, validate=validate)
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
):
    df = pl.read_csv(input)
    r = omega_prime.Recording(df, validate=validate)
    if odr is not None:
        r.map = omega_prime.MapOdr.from_file(odr)
    r.to_mcap(output)


@app.command(help="Check an omega-prime file for specification conformance.")
def validate(
    input: Annotated[Path, typer.Argument(help="Path to omega file to validate", exists=True, dir_okay=False)],
):
    omega_prime.Recording.from_file(input, validate=True, parse_map=True)
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
    if Path(input).suffix == ".mcap":
        with Path(input).open("rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            gm = next(reader.iter_decoded_messages(topics="/ground_truth_map"))
        map = omega_prime.MapOdr.create(odr_xml=gm[3].open_drive_xml, name="", step_size=0.01)
        map.to_file(output)
        return None

    r = omega_prime.Recording.from_file(input, validate=False)
    if isinstance(r.map, omega_prime.MapOdr):
        r.map.to_file(output)
    else:
        raise ValueError("The provided omega-prime file does not contain a map in ASAM OpenDRIVE format.")


@app.command(help="Converts MCAP or OSI omega-prime into a parquet file (beneficial for faster loading of large data).")
def to_parquet(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Path to the omega-prime mcap file.")],
    output: Annotated[
        Path | None,
        typer.Argument(help="Where to store the parquet file. If None or directory, stored filename will be used."),
    ] = None,
):
    r = omega_prime.Recording.from_file(input, validate=False, parse_map=False)
    if output is None:
        output = Path(input).parent / f"{Path(input).stem}.parquet"
    r.to_parquet(output)


@app.command(help="Visualize an omega-prime recording using Altair. Opens interactive plot in browser.")
def visualize(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    height: int = 500,
    width: int = 1600,
    start_frame: int = 0,
    end_frame: int = -1,
    plot_map: bool = True,
    use_vegafusion: bool = False,
):
    if use_vegafusion:
        alt.data_transformers.enable("vegafusion")
    alt.renderers.enable("browser")
    r = omega_prime.Recording.from_file(input, validate=False, parse_map=True)
    r.plot_altair(start_frame=start_frame, end_frame=end_frame, height=height, width=width, plot_map=plot_map).show()


@app.command(help="Attach an ASAM OpenDRIVE (.xodr) map to an existing omega-prime recording and write a new file.")
def attach_odr(
    input: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Path to existing omega-prime file (.mcap or .parquet)")
    ],
    odr: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Path to ASAM OpenDRIVE .xodr file")],
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output path (.mcap or .parquet). If not provided, uses '<input>_with_odr.<ext>' in the same directory."
        ),
    ] = None,
):
    r = omega_prime.Recording.from_file(input, validate=False, parse_map=False)
    r.map = omega_prime.MapOdr.from_file(odr)

    if output is None:
        output = input.parent / f"{input.stem}_with_odr{input.suffix}"

    if Path(output).suffix == ".parquet":
        r.to_parquet(output)
    else:
        r.to_mcap(output)


def main():
    load_converters_into_cli(app)
    app()


if __name__ == "__main__":
    main()
