import json
from pathlib import Path

import omega_prime

p = Path(__file__).parent.parent / "example_files/"
with open(p / "mapping.json") as f:
    mapping = json.load(f)


def test_projection():
    rec = omega_prime.Recording.from_file(p / mapping[0][0], p / mapping[0][1])

    rec.apply_projections()


if __name__ == "__main__":
    test_projection()
    # pass
