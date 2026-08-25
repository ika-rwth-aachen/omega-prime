"""."""

from omega_prime.metrics.qualification.cli.mov_obj_type import MovObjTypeCli


def test_get_default() -> None:
    assert MovObjTypeCli.get_default() == ("pedestrian", "vehicle")
