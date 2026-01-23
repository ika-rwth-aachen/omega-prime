"""."""
from pathlib import Path

import pytest


@pytest.fixture
def files_dir() -> Path:
    return Path(__file__).parent.parent / "example_files/"

