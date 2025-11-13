import numpy as np
import omega_prime
import cProfile
from pathlib import Path
from pstats import Stats


def test_map_segmentation(
    file=Path(__file__).parent.parent / "example_files/osi_centerline_example.mcap",
):
    with cProfile.Profile() as pr:
        output_path = Path("test_map_segmentation_output")
        output_path.mkdir(parents=True, exist_ok=True)

        r = omega_prime.Recording.from_file(filepath=file, split_lanes=True, split_lanes_length=15)
        r.create_mapsegments()
        mapsegment = r.mapsegment

        id = r.host_vehicle_idx
        positions = None

        if id is not None:
            # Assuming r.moving_objects is a list of objects with x and y attributes
            x_values = r.moving_objects[id].x.to_numpy()  # Convert Polars Series to NumPy array
            y_values = r.moving_objects[id].y.to_numpy()  # Convert Polars Series to NumPy array
            frame = r.moving_objects[id]._df["frame"].to_numpy()  # Convert Polars Series to NumPy array

            # Combine x and y into a single NumPy array of shape (n, 3)
            positions = np.column_stack((frame, x_values, y_values))

            segments = mapsegment.trajectory_segment_detection(positions)  # noqa: F841

        mapsegment.plot(
            output_plot=output_path,
            trajectory=positions,
            plot_lane_ids=False,
            plot_intersection_polygons=True,
            plot_connection_polygons=False,
        )

    stats = Stats(pr)
    stats.dump_stats("test_map_segmentation.prof")


if __name__ == "__main__":
    test_map_segmentation()
