"""Parameter-string parsing helpers shared by the planner node and the debug tool."""

import re


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "prompt"


def parse_float_csv(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def parse_vector3_parameter(value: str, name: str) -> list[float]:
    values = parse_float_csv(value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated floats")
    return values
