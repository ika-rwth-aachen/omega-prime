from omega_prime.cli import app
from omega_prime.metrics.qualification.cli.cmd_line_interface import qualification_cli


def test_is_qualification_command_present() -> None:
    assert any(cmd.callback == qualification_cli.entry_point for cmd in app.registered_commands)