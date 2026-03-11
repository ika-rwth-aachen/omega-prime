"""."""

from omega_prime.metrics.qualification.english_syntax import format_items, get_is_ending


def test_format_items() -> None:
    assert format_items([]) == ""
    assert format_items(["1"]) == "1"
    assert format_items(["1", "2"]) == "1 and 2"
    assert format_items(["1", "2", "3"]) == "1, 2 and 3"


def test_get_is_ending() -> None:
    assert get_is_ending(0) == ("are", "s")
    assert get_is_ending(1) == ("is", "")
    assert get_is_ending(2) == ("are", "s")
