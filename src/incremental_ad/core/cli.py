from argparse import ArgumentTypeError, Namespace
from typing import Any


def str_to_bool(v: str) -> bool:
    """Argparse type for explicit boolean flags ('true' / 'false')."""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    raise ArgumentTypeError(f"Expected 'true' or 'false', got '{v}'")


def pluck(args: Namespace, prefix: str) -> dict[str, Any]:
    """Extract argparse attributes whose name starts with `<prefix>_`.

    Returns a dict mapping the un-prefixed name to its value. Useful for
    splatting into a dataclass whose field names match the un-prefixed args:
        SWaTConfig(**pluck(args, "swat"))

    Example: with args.swat_window_len=100, args.swat_stride=1, then
    `pluck(args, "swat")` returns {"window_len": 100, "stride": 1}.
    """
    p = f"{prefix}_"
    return {k.removeprefix(p): v for k, v in vars(args).items() if k.startswith(p)}
