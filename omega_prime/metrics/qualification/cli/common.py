"""."""


def to_osi_str(cli_option: str) -> str:
    return "_".join(cli_option.upper().split("-"))
