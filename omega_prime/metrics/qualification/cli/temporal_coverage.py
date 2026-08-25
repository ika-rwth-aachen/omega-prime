"""."""

from datetime import datetime

import typer
import typer.models


H_EXPECTED_START = "Expected start time in ISO 8601 format for temporal coverage."
H_EXPECTED_END = "Expected end time in ISO 8601 format for temporal coverage."


class TemporalCoverageCli:
    @staticmethod
    def get_expected_start_option() -> typer.models.OptionInfo:
        return typer.Option("--expected-start", help=H_EXPECTED_START)

    @staticmethod
    def get_expected_end_option() -> typer.models.OptionInfo:
        return typer.Option("--expected-end", help=H_EXPECTED_END)

    @staticmethod
    def build_kwargs(expected_start: datetime | None, expected_end: datetime | None) -> dict[str, object]:
        if expected_start is None or expected_end is None:
            raise ValueError("expected_start and expected_end must be provided for temporal-coverage")
        return {
            "expected_start": expected_start,
            "expected_end": expected_end,
        }
