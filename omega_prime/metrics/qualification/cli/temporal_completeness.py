"""."""

import typer
import typer.models


H_TIMING_TOLERANCE = "Allowed relative RMS timing deviation per track, e.g. 0.05 for 5%."


class TemporalCompletenessCli:
    @staticmethod
    def get_timing_tolerance_option() -> typer.models.OptionInfo:
        return typer.Option("--timing-tolerance", help=H_TIMING_TOLERANCE)

    @staticmethod
    def build_kwargs(expected_frequency: float, timing_tolerance: float) -> dict[str, object]:
        return {
            "expected_frequency": expected_frequency,
            "timing_tolerance": timing_tolerance,
        }
