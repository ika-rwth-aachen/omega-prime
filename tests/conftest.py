"""."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def files_dir() -> Path:
    return Path(__file__).parent / "files/"
