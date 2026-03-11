"""."""

from pathlib import Path

import pytest

import omega_prime


@pytest.fixture(scope="session")
def cut_in(files_dir: Path):
    return omega_prime.Recording.from_file(str(files_dir / "alks_cut-in.osi"), str(files_dir / "straight_500m.xodr"))