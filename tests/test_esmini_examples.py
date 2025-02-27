import omega_format
import json
from pathlib import Path

def test_esmini_examples():
    p = Path('example_files/')
    with open(p/'mapping.json') as f:
        mapping = json.load(f)[:2]

    for p_osi, p_odr in mapping:
        rec = omega_format.Recording.from_file(p/p_osi, p/p_odr)
        rec.to_mcap(f'{Path(p_osi).stem}.mcap')
        rec = omega_format.Recording.from_file(f'{Path(p_osi).stem}.mcap')