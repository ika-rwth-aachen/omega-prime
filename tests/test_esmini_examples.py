import omega_format
import json
from pathlib import Path
import cProfile
from pstats import Stats


def test_esmini_examples():
    p = Path('example_files/')
    with open(p/'mapping.json') as f:
        mapping = json.load(f)[:]

    with cProfile.Profile() as pr:
        for p_osi, p_odr in mapping:
            rec = omega_format.Recording.from_file(p/p_osi, p/p_odr)
            rec.to_mcap(f'{Path(p_osi).stem}.mcap')
            rec = omega_format.Recording.from_file(f'{Path(p_osi).stem}.mcap')
        stats = Stats(pr)
    stats.dump_stats("test.prof")
if __name__ == "__main__":
    test_esmini_examples()