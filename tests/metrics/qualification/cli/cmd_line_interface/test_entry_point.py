"""."""

from pathlib import Path

from omega_prime.metrics.qualification.cli.cmd_line_interface import qualification_cli
from omega_prime.metrics.qualification.cli.metric_dispatch import CliMetric


def test_entry_point(files_dir: Path) -> None:
    qualification_cli.entry_point(files_dir / "pedestrian.osi", [CliMetric.CLASS_COMPLETENESS])
