import cProfile
import json
from pathlib import Path
from pstats import Stats

import omega_prime

p = Path("example_files/")
with open(p / "mapping.json") as f:
    mapping = json.load(f)


def test_validation():
    with cProfile.Profile() as pr:
        rec = omega_prime.Recording.from_file(p / mapping[3][0], p / mapping[3][1])
        rec.to_mcap("validated.mcap")
        rec = omega_prime.Recording.from_file("validated.mcap")
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


if __name__ == "__main__":
    test_validation()
    test_esmini_examples()
    test_validation()
