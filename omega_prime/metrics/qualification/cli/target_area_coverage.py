"""."""

import click
import typer
import typer.models


H_EXPECTED_AREA_COORD = "Expected target area coordinate as two floats. Repeat the option at least three times."
H_EXPECTED_AREA_SOURCE_CRS = "CRS of expected target area coordinates, e.g. EPSG:4326."


class TargetAreaCoverageCli:
    @staticmethod
    def get_expected_area_coord_option() -> typer.models.OptionInfo:
        return typer.Option(
            "--expected-area-coord",
            help=H_EXPECTED_AREA_COORD,
            click_type=click.Tuple([float, float]),
        )

    @staticmethod
    def get_expected_area_source_crs_option() -> typer.models.OptionInfo:
        return typer.Option("--expected-area-source-crs", help=H_EXPECTED_AREA_SOURCE_CRS)

    @staticmethod
    def normalize_area_coords(values: list[object]) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for value in values:
            x, y = value
            coords.append((float(x), float(y)))
        return coords

    @classmethod
    def _offset_to_tuple(cls, offset: object | None) -> tuple[float, float, float] | tuple[()]:
        if offset is None:
            return ()
        return (
            float(getattr(offset, "x")),
            float(getattr(offset, "y")),
            float(getattr(offset, "yaw")),
        )

    @classmethod
    def _get_recording_projection(cls, recording: object) -> tuple[str | None, tuple[float, float, float] | tuple[()]]:
        projections = getattr(recording, "projections", [])
        for projection in projections:
            proj_string = projection.get("proj_string")
            if proj_string:
                return proj_string, cls._offset_to_tuple(projection.get("offset"))
        return None, ()

    @classmethod
    def _get_map_projection(cls, recording: object) -> tuple[str | None, tuple[float, float, float] | tuple[()]]:
        map_obj = getattr(recording, "map", None)
        if map_obj is None:
            return None, ()

        proj_string = getattr(map_obj, "proj_string", None)
        if not proj_string and hasattr(map_obj, "parse"):
            map_obj.parse()
            proj_string = getattr(map_obj, "proj_string", None)

        return proj_string, cls._offset_to_tuple(getattr(map_obj, "proj_offset", None))

    @classmethod
    def resolve_dataset_projection(
        cls,
        recording: object,
    ) -> tuple[str | None, tuple[float, float, float] | tuple[()]]:
        proj_string, offset = cls._get_recording_projection(recording)
        if proj_string:
            return proj_string, offset
        return cls._get_map_projection(recording)

    @classmethod
    def build_kwargs(
        cls,
        recording: object,
        expected_area_coords: list[object],
        expected_area_source_crs: str | None,
    ) -> dict[str, object]:
        if not expected_area_coords:
            raise ValueError("expected_area_coords must be provided for target-area-coverage")

        metric_kwargs: dict[str, object] = {
            "expected_area_coords": cls.normalize_area_coords(expected_area_coords),
        }
        if expected_area_source_crs is not None:
            metric_kwargs["expected_area_source_crs"] = expected_area_source_crs
            if expected_area_source_crs != "":
                dataset_proj4, dataset_frame_offset = cls.resolve_dataset_projection(recording)
                if not dataset_proj4:
                    raise ValueError(
                        "dataset projection could not be derived from recording or map for target-area-coverage"
                    )
                metric_kwargs["dataset_proj4"] = dataset_proj4
                if dataset_frame_offset:
                    metric_kwargs["dataset_frame_offset"] = dataset_frame_offset

        return metric_kwargs
