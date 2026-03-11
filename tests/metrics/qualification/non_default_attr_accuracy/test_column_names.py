from omega_prime.qualification.non_default_attr_accuracy import get_column_names


def test_get_column_names() -> None:
    assert get_column_names(int) == ["total_nanos", "idx", "type", "role", "subtype"]
    # fmt: off
    ref = [
        "x", "y", "z",
        "vel_x", "vel_y", "vel_z",
        "acc_x", "acc_y", "acc_z",
        "length", "width", "height",
        "roll", "pitch", "yaw"
    ]
    # fmt: on
    assert get_column_names(float) == ref
