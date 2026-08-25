"""."""

from .mov_obj_type import MovObjTypeCli


class ObjectTypeCoverageCli:
    @staticmethod
    def build_kwargs(types: list[str]) -> dict[str, object]:
        return {
            "expected_types": MovObjTypeCli.to_osi_list(types),
        }
