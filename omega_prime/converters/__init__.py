from .lxd import LxdConverter
import typer

from importlib.metadata import entry_points
from warnings import warn

__all__ = ["app"]

app = typer.Typer(pretty_exceptions_show_locals=False)

app.command("from-lxd", help="Convert datasets from LevelXData to omega-prime.")(LxdConverter.convert_cli)


def load_converters_into_cli(app):
    discovered_plugins = {o.name: o for o in entry_points(group="omega_prime.plugins.converter")}
    for p_name, p in discovered_plugins.items():
        try:
            converter = p.load()
        except ModuleNotFoundError:
            warn(f"Failed to load converter extension from dist `{p.dist.name}` called `{p.name}`.")
        try:
            app.command(p.name, help=f"Convert datasets with {p.dist.name} {p.name} to omega-prime.")(
                converter.convert_cli
            )
        except AttributeError:
            warn(f"Converter `{p.name}` of `{p.dist.name}` is not a `omega_prime.DatasetConverter`")
