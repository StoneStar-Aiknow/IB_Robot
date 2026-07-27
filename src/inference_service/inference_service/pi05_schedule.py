"""Strict PI0.5 velocity denoising schedule representation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from inference_manifest.json_utils import load_json_strict

PI05_SCHEDULE_FORMAT = "pi05-denoising-schedule-v1"
PI05_SCHEDULE_ALGORITHM = "euler"
PI05_MODEL_OUTPUT = "velocity"

_SCHEDULE_KEYS = frozenset({"format", "name", "algorithm", "model_output", "timesteps"})


@dataclass(frozen=True)
class PI05DenoisingSchedule:
    """Validated Euler timesteps for an AE that returns velocity."""

    name: str
    timesteps: tuple[float, ...]
    format: str = PI05_SCHEDULE_FORMAT
    algorithm: str = PI05_SCHEDULE_ALGORITHM
    model_output: str = PI05_MODEL_OUTPUT

    def __post_init__(self) -> None:
        if self.format != PI05_SCHEDULE_FORMAT:
            raise ValueError(f"schedule format must be {PI05_SCHEDULE_FORMAT!r}")
        if type(self.name) is not str or not self.name.strip() or self.name != self.name.strip():
            raise ValueError("schedule name must be a non-empty, trimmed string")
        if self.algorithm != PI05_SCHEDULE_ALGORITHM:
            raise ValueError(f"schedule algorithm must be {PI05_SCHEDULE_ALGORITHM!r}")
        if self.model_output != PI05_MODEL_OUTPUT:
            raise ValueError(f"schedule model_output must be {PI05_MODEL_OUTPUT!r}")
        if type(self.timesteps) is not tuple:
            raise ValueError("schedule timesteps must be a tuple")
        if len(self.timesteps) < 2:
            raise ValueError("schedule timesteps must contain at least 2 points")
        if any(type(value) is not float or not math.isfinite(value) for value in self.timesteps):
            raise ValueError("schedule timesteps must contain only finite numbers")
        if self.timesteps[0] != 1.0:
            raise ValueError("schedule timesteps must start at 1.0")
        if self.timesteps[-1] != 0.0:
            raise ValueError("schedule timesteps must end at 0.0")
        if any(current <= following for current, following in pairwise(self.timesteps)):
            raise ValueError("schedule timesteps must be strictly decreasing")

    @property
    def step_count(self) -> int:
        return len(self.timesteps) - 1

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "name": self.name,
            "algorithm": self.algorithm,
            "model_output": self.model_output,
            "timesteps": list(self.timesteps),
        }


def parse_pi05_schedule(value: object, *, source: str = "schedule") -> PI05DenoisingSchedule:
    """Parse an already-decoded schedule while rejecting coercion and unknown fields."""

    if type(value) is not dict:
        raise ValueError(f"{source} must be a JSON object")
    unknown = sorted(set(value) - _SCHEDULE_KEYS)
    missing = sorted(_SCHEDULE_KEYS - set(value))
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise ValueError(f"invalid {source}: {', '.join(details)}")

    timesteps = value["timesteps"]
    if type(timesteps) is not list:
        raise ValueError(f"{source} timesteps must be a JSON array")
    if any(type(item) not in (int, float) or type(item) is bool for item in timesteps):
        raise ValueError(f"{source} timesteps must contain only numbers")
    try:
        return PI05DenoisingSchedule(
            format=value["format"],
            name=value["name"],
            algorithm=value["algorithm"],
            model_output=value["model_output"],
            timesteps=tuple(float(item) for item in timesteps),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {source}: {exc}") from exc


def load_pi05_schedule(path: str | Path) -> PI05DenoisingSchedule:
    schedule_path = Path(path).expanduser().resolve(strict=True)
    return parse_pi05_schedule(load_json_strict(schedule_path), source=str(schedule_path))


def uniform_pi05_schedule(num_inference_steps: int, *, name: str = "uniform") -> PI05DenoisingSchedule:
    if type(num_inference_steps) is not int or num_inference_steps < 1:
        raise ValueError("num_inference_steps must be a positive integer")
    timesteps = tuple(1.0 - step / num_inference_steps for step in range(num_inference_steps)) + (0.0,)
    return PI05DenoisingSchedule(name=name, timesteps=timesteps)


def write_pi05_schedule(schedule: PI05DenoisingSchedule, path: str | Path) -> Path:
    if not isinstance(schedule, PI05DenoisingSchedule):
        raise TypeError("schedule must be a PI05DenoisingSchedule")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(schedule.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination


__all__ = [
    "PI05DenoisingSchedule",
    "PI05_MODEL_OUTPUT",
    "PI05_SCHEDULE_ALGORITHM",
    "PI05_SCHEDULE_FORMAT",
    "load_pi05_schedule",
    "parse_pi05_schedule",
    "uniform_pi05_schedule",
    "write_pi05_schedule",
]
