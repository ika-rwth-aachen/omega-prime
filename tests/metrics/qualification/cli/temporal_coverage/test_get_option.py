"""."""

import typer.models

from omega_prime.metrics.qualification.cli.temporal_coverage import (
    H_EXPECTED_END,
    H_EXPECTED_START,
    TemporalCoverageCli,
)


def test_get_expected_start_option() -> None:
    typer_option = TemporalCoverageCli.get_expected_start_option()
    assert isinstance(typer_option, typer.models.OptionInfo)
    assert typer_option.help == H_EXPECTED_START
    assert typer_option.default == "--expected-start"
    assert typer_option.param_decls == ()
    assert typer_option.click_type is None


def test_get_expected_end_option() -> None:
    typer_option = TemporalCoverageCli.get_expected_end_option()
    assert isinstance(typer_option, typer.models.OptionInfo)
    assert typer_option.help == H_EXPECTED_END
    assert typer_option.default == "--expected-end"
    assert typer_option.param_decls == ()
    assert typer_option.click_type is None
