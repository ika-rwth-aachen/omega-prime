import betterosi


version_major, version_minor, version_patch = [int(o) for o in betterosi.version.split(".")]

__all__ = [
    "LaneBoundaryType",
    "LaneType",
    "LaneSubtype",
    "MovingObjectType",
    "MovingObjectSubtype",
    "MovingObjectRole",
]

LaneBoundaryType = betterosi.LaneBoundary.Classification.Type
LaneType = betterosi.Lane.Classification.Type
LaneSubtype = betterosi.Lane.Classification.Subtype
MovingObjectType = betterosi.MovingObject.Type
MovingObjectSubtype = betterosi.MovingObject.VehicleClassification.Type
MovingObjectRole = betterosi.MovingObject.VehicleClassification.Role
