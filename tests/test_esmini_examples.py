import cProfile
import json
import numpy as np
import shapely
from pathlib import Path
from pstats import Stats


import omega_prime

p = Path(__file__).parent.parent / "example_files/"
with open(p / "mapping.json") as f:
    mapping = json.load(f)


def test_validation():
    with cProfile.Profile() as pr:
        rec = omega_prime.Recording.from_file(p / mapping[3][0], p / mapping[3][1])
        rec.to_mcap("validated.mcap")
        rec = omega_prime.Recording.from_file("validated.mcap", validate=True)
        stats = Stats(pr)
    stats.dump_stats("test_validate.prof")


def test_esmini_examples():
    with cProfile.Profile() as pr:
        for p_osi, p_odr in mapping:
            rec = omega_prime.Recording.from_file(p / p_osi, p / p_odr, validate=False)
            rec.to_mcap(f"{Path(p_osi).stem}.mcap")
            rec = omega_prime.Recording.from_file(f"{Path(p_osi).stem}.mcap", validate=False)
        stats = Stats(pr)
    stats.dump_stats("test.prof")


def test_interpolate():
    rec = omega_prime.Recording.from_file(p / mapping[3][0], p / mapping[3][1], validate=False)
    rec.interpolate(hz=10)

def test_centerline():
    with cProfile.Profile() as pr:
        # Load the recording
        # rec = omega_prime.Recording.from_file(p / mapping[3][0], p / mapping[3][1], validate=False)
        rec = omega_prime.Recording.from_file("example_files/01_tracks.mcap")
        # Create a locator and remove all polygons from the lanes
        locator = omega_prime.Locator.from_mapodr(rec.map)
        for lane in locator.all_lanes:
            lane.polygon = None  # Remove polygons
        
        # Create a new locator using only centerlines
        locator_cl = omega_prime.Locator(locator.all_lanes, locator.internal2external_laneid)
        
        # Prepare input coordinates for xys2sts
        xys = np.stack([rec.moving_objects[0].x, rec.moving_objects[0].y]).T
        
        # Run xys2sts and ensure it executes without errors
        sts_cl = locator_cl.xys2sts(xys)
        
        # Check that str_tree.geometries only contains LineString objects
        assert all(isinstance(geom, shapely.LineString) for geom in locator_cl.str_tree.geometries), \
            "str_tree contains non-LineString geometries"
        
        #check that sts_cl is not empty or has expected properties
        assert len(sts_cl["s"]) > 0, "sts_cl output is empty"
        assert len(sts_cl["t"]) > 0, "sts_cl output is empty"
    
    # Save profiling results to a file
    stats = Stats(pr)
    stats.dump_stats("test_centerline.prof")
    

def test_locator():
    with cProfile.Profile() as pr:
        # Load the recording
        # rec = omega_prime.Recording.from_file(p / mapping[3][0], p / mapping[3][1], validate=False)
        rec = omega_prime.Recording.from_file("example_files/01_tracks.mcap")
        # Create a locator and remove all polygons from the lanes
        locator = omega_prime.Locator.from_mapodr(rec.map)

        
        # Prepare input coordinates for xys2sts
        xys = np.stack([rec.moving_objects[0].x, rec.moving_objects[0].y]).T
        
        # Run xys2sts and ensure it executes without errors
        sts_cl = locator.xys2sts(xys)
        
        # Check that str_tree.geometries only contains LineString objects
        assert all(isinstance(geom, shapely.LineString) for geom in locator.str_tree.geometries), \
            "str_tree contains non-LineString geometries"
        
        #check that sts_cl is not empty or has expected properties
        assert len(sts_cl["s"]) > 0, "sts_cl output is empty"
        assert len(sts_cl["t"]) > 0, "sts_cl output is empty"
    
    # Save profiling results to a file
    stats = Stats(pr)
    stats.dump_stats("test_centerline.prof")
    
def test_parquet():
    rec = omega_prime.Recording.from_file(p / mapping[3][0], p / mapping[3][1])
    rec.to_parquet("test.parquet")
    rec = omega_prime.Recording.from_file("test.parquet")
    assert rec.map is not None


if __name__ == "__main__":
    pass
