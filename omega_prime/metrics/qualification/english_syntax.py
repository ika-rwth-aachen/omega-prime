from collections.abc import Sequence


def get_is_ending(n: int) -> tuple[str, str]:
    return ("is", "") if n == 1 else ("are", "s")


def format_items(items: Sequence[str]) -> str:
    num_items = len(items)
    if num_items in (0, 1):
        return ", ".join(items)
    else:
        return f"{', '.join(items[:-1])} and {items[-1]}"
