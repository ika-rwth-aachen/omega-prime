"""."""

import typer.models

from omega_prime.metrics.qualification.cli.metric_dispatch import CliMetric


def test_get_option() -> None:
    typer_option = CliMetric.get_option()
    assert isinstance(typer_option, typer.models.OptionInfo)
    assert typer_option.help == "Select one or more metrics."
    assert typer_option.default == "--metric"
    assert typer_option.param_decls == ("-m",)
    assert typer_option.click_type is None
