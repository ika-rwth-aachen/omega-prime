"""."""


class TemporalCompletenessCli:
    @staticmethod
    def build_kwargs(expected_frequency: float) -> dict[str, object]:
        return {
            "expected_frequency": expected_frequency,
        }
