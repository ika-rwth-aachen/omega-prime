"""."""


class NonDefaultAttributesAccuracyCli:
    @staticmethod
    def build_kwargs(columns: list[str]) -> dict[str, object]:
        return {
            "columns": columns,
        }
