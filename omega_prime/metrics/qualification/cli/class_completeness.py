"""."""

from .mov_obj_type import MovObjTypeCli
from .vehicle_role import VehicleRoleCli
from .vehicle_type import VehicleTypeCli


class ClassCompletenessCli:
    @staticmethod
    def build_kwargs(types: list[str], subtypes: list[str], roles: list[str]) -> dict[str, object]:
        return {
            "expected_types": MovObjTypeCli.to_osi_list(types),
            "expected_subtypes": VehicleTypeCli.to_osi_list(subtypes),
            "expected_roles": VehicleRoleCli.to_osi_list(roles),
        }
