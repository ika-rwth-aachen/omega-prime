"""."""

import typer.models

from omega_prime.metrics.qualification.cli.temporal_completeness import (
    H_TIMING_TOLERANCE,
    TemporalCompletenessCli,
)


def test_get_timing_tolerance_option() -> None:
    typer_option = TemporalCompletenessCli.get_timing_tolerance_option()
    assert isinstance(typer_option, typer.models.OptionInfo)
    assert typer_option.help == H_TIMING_TOLERANCE
    assert typer_option.default == "--timing-tolerance"
    assert typer_option.param_decls == ()
    assert typer_option.click_type is None
