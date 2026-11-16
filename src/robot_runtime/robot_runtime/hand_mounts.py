"""Explicit hand mounting shared by configuration and robot-specific consumers."""

HAND_SIDES = ("left", "right")


def installed_hands(value) -> tuple[str, ...]:
    """Validate mounting declarations; missing feedback never implies removal."""
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ValueError(f"installed_hands must be a list of sides from {HAND_SIDES}, got {value!r}")
    if any(not isinstance(side, str) for side in value):
        raise ValueError(f"installed_hands must name sides from {HAND_SIDES}, got {value!r}")
    if len(set(value)) != len(value) or not set(value) <= set(HAND_SIDES):
        raise ValueError(f"installed_hands must name each of {HAND_SIDES} at most once; got {value!r}")
    return tuple(side for side in HAND_SIDES if side in value)
