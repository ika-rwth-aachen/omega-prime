"""."""

from pathlib import Path

import pytest

import omega_prime


@pytest.fixture()
def rec(files_dir: Path):
    return omega_prime.Recording.from_file(str(files_dir / "pedestrian.osi"))
