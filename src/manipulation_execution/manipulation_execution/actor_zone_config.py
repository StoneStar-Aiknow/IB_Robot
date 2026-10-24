"""Read and check the two files of the standing-zone actor selector.

``actor_zone.yaml`` says where on the floor the actor stands: four image points
of a floor trapezoid, and the horizontal band of the image people are looked
for in. ``actor_tuning.yaml`` holds the selector's counts, weights and
thresholds; it is optional and every key in it is optional. Both are read once
when the node starts, so editing either takes effect on the next launch with
no rebuild.

A wrong file is an error, never a silent fallback to whole-frame selection: a
robot that starts with the wrong zone looks like it works while picking the
wrong people, which is harder to notice than a robot that refuses to start.
Every error names the file and the field.

The launch builder and the node both call ``load_zone`` / ``load_tuning``, and
so does the command-line check the run script and the zone tools use::

    python3 -m manipulation_execution.actor_zone_config --zone F [--tuning F]

It prints ``OK`` and each file's sha256 and exits 0, or prints the offending
field and exits 1. Only the standard library and yaml are imported here, so the
builder can import this without leaving its import budget.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

ZONE_VERSION = 1
TUNING_VERSION = 1
QUAD_CORNERS = ("near_left", "near_right", "far_right", "far_left")
# A person whose feet are cut off by the bottom of the frame has a box bottom
# 957-967 px on the 972 px half stream (10-24 records), so the near edge must
# stay more than 15 px above the bottom for "feet out of frame" to read as
# "outside the zone".
MIN_NEAR_EDGE_MARGIN_PX = 16
DEFAULT_NEAR_EDGE_MARGIN_PX = 30
# mirror_session's cancel_confirm_s. A shorter hold may let the session go on
# mirroring the next person without passing through STANDBY.
SESSION_CANCEL_CONFIRM_S = 2.0

_ZONE_REQUIRED = ("version", "image_size", "crop_x", "floor_quad")
_ZONE_OPTIONAL = ("near_edge_margin_px", "note")


class ZoneConfigError(ValueError):
    """A zone or tuning file that must not be used. The message names the field."""


@dataclass(frozen=True)
class ZoneConfig:
    path: str
    sha256: str
    image_size: tuple[int, int]
    crop_x: tuple[float, float]
    # near_left, near_right, far_right, far_left, in image pixels.
    floor_quad: tuple[tuple[float, float], ...]
    near_edge_margin_px: int
    note: str


@dataclass(frozen=True)
class TuningConfig:
    acquire_updates: int = 2
    release_updates: int = 3
    release_hold_s: float = 3.0
    track_keep_updates: int = 2
    foot_half_width: float = 0.12
    center_weight: float = 0.6
    front_weight: float = 0.4
    # None turns the occlusion rule off.
    occlude_overlap: float | None = 0.3
    occlude_dy: float = 0.05
    # Optional association gates; None means no gate.
    gate_dx: float | None = None
    gate_dy: float | None = None
    gate_h_ratio: float | None = None
    # Where the values came from: None for the built-in defaults.
    path: str | None = None
    sha256: str | None = None


# key: (kind, low, low_inclusive, high, high_inclusive, nullable)
_TUNING_RULES = {
    "acquire_updates": ("int", 1, True, 10, True, False),
    "release_updates": ("int", 1, True, 10, True, False),
    "release_hold_s": ("float", 0.0, True, 30.0, True, False),
    "track_keep_updates": ("int", 0, True, 10, True, False),
    "foot_half_width": ("float", 0.02, True, 0.5, True, False),
    "center_weight": ("float", 0.0, True, None, False, False),
    "front_weight": ("float", 0.0, True, None, False, False),
    "occlude_overlap": ("float", 0.0, False, 1.0, True, True),
    "occlude_dy": ("float", 0.0, True, None, False, False),
    "gate_dx": ("float", 0.0, False, None, False, True),
    "gate_dy": ("float", 0.0, False, None, False, True),
    "gate_h_ratio": ("float", 1.0, False, None, False, True),
}


def _read(path: str | Path, what: str) -> tuple[str, str, object]:
    file = Path(path)
    try:
        raw = file.read_bytes()
    except OSError as exc:
        raise ZoneConfigError(f"{file}: cannot read {what} file: {exc}") from exc
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ZoneConfigError(f"{file}: {what} file is not valid yaml: {exc}") from exc
    return str(file), hashlib.sha256(raw).hexdigest(), data


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(path: str, field: str, value: object) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ZoneConfigError(f"{path}: {field}: must be a finite number, got {value!r}")
    return float(value)


def _pair(path: str, field: str, value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ZoneConfigError(f"{path}: {field}: must be a list of two numbers, got {value!r}")
    return _number(path, f"{field}[0]", value[0]), _number(path, f"{field}[1]", value[1])


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def load_zone(path: str | Path) -> ZoneConfig:
    """Read and fully check ``actor_zone.yaml``; raise ``ZoneConfigError`` naming the field."""
    path, digest, data = _read(path, "zone")
    if not isinstance(data, dict):
        raise ZoneConfigError(f"{path}: zone file must be a mapping, got {type(data).__name__}")
    unknown = sorted(str(key) for key in data if key not in _ZONE_REQUIRED + _ZONE_OPTIONAL)
    if unknown:
        raise ZoneConfigError(f"{path}: {unknown[0]}: unknown key (allowed: {', '.join(_ZONE_REQUIRED + _ZONE_OPTIONAL)})")
    for key in _ZONE_REQUIRED:
        if key not in data:
            raise ZoneConfigError(f"{path}: {key}: missing")

    version = data["version"]
    if not _is_number(version) or isinstance(version, float) or version != ZONE_VERSION:
        raise ZoneConfigError(f"{path}: version: must be {ZONE_VERSION}, got {version!r}")

    size = data["image_size"]
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or not all(_is_number(v) and not isinstance(v, float) and v > 0 for v in size)
    ):
        raise ZoneConfigError(f"{path}: image_size: must be two positive integers [width, height], got {size!r}")
    width, height = int(size[0]), int(size[1])

    left, right = _pair(path, "crop_x", data["crop_x"])
    if not 0.0 <= left < right <= width:
        raise ZoneConfigError(f"{path}: crop_x: must satisfy 0 <= L < R <= {width}, got [{left:g}, {right:g}]")

    quad_raw = data["floor_quad"]
    if not isinstance(quad_raw, (list, tuple)) or len(quad_raw) != 4:
        raise ZoneConfigError(f"{path}: floor_quad: must be 4 points ({', '.join(QUAD_CORNERS)}), got {quad_raw!r}")
    quad = tuple(_pair(path, f"floor_quad.{name}", point) for name, point in zip(QUAD_CORNERS, quad_raw))
    for name, (x, y) in zip(QUAD_CORNERS, quad):
        if not (0.0 <= x <= width and 0.0 <= y <= height):
            raise ZoneConfigError(f"{path}: floor_quad.{name}: ({x:g}, {y:g}) is outside the {width}x{height} image")
        if not left <= x <= right:
            raise ZoneConfigError(f"{path}: floor_quad.{name}: x {x:g} is outside crop_x [{left:g}, {right:g}]")
    near_left, near_right, far_right, far_left = quad
    if not (near_left[0] < near_right[0] and far_left[0] < far_right[0]):
        raise ZoneConfigError(f"{path}: floor_quad: left points must be left of right points (order is {', '.join(QUAD_CORNERS)})")
    if not (near_left[1] > far_left[1] and near_right[1] > far_right[1]):
        raise ZoneConfigError(f"{path}: floor_quad: near points must be below far points (order is {', '.join(QUAD_CORNERS)})")
    crosses = [_cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]) for i in range(4)]
    if not (all(c > 0.0 for c in crosses) or all(c < 0.0 for c in crosses)):
        raise ZoneConfigError(f"{path}: floor_quad: the 4 points must form a convex quadrilateral (no dents, crossings or 3 points on a line)")

    margin = data.get("near_edge_margin_px", DEFAULT_NEAR_EDGE_MARGIN_PX)
    if not _is_number(margin) or isinstance(margin, float) or margin < MIN_NEAR_EDGE_MARGIN_PX:
        raise ZoneConfigError(f"{path}: near_edge_margin_px: must be an integer >= {MIN_NEAR_EDGE_MARGIN_PX}, got {margin!r}")
    for name, (_, y) in (("near_left", near_left), ("near_right", near_right)):
        if y > height - margin:
            raise ZoneConfigError(
                f"{path}: floor_quad.{name}: y {y:g} is closer than near_edge_margin_px {margin} to the image bottom {height}"
            )

    note = data.get("note", "")
    if note is None:
        note = ""
    if not isinstance(note, str):
        raise ZoneConfigError(f"{path}: note: must be a string")

    return ZoneConfig(
        path=path,
        sha256=digest,
        image_size=(width, height),
        crop_x=(left, right),
        floor_quad=quad,
        near_edge_margin_px=int(margin),
        note=note,
    )


def load_tuning(path: str | Path | None) -> tuple[TuningConfig, list[str]]:
    """Read ``actor_tuning.yaml`` (None or "" means all defaults); return it and any warnings."""
    if path is None or str(path) == "":
        return TuningConfig(), []
    path, digest, data = _read(path, "tuning")
    if not isinstance(data, dict):
        raise ZoneConfigError(f"{path}: tuning file must be a mapping, got {type(data).__name__}")
    unknown = sorted(str(key) for key in data if key != "version" and key not in _TUNING_RULES)
    if unknown:
        raise ZoneConfigError(f"{path}: {unknown[0]}: unknown key")
    version = data.get("version")
    if not _is_number(version) or isinstance(version, float) or version != TUNING_VERSION:
        raise ZoneConfigError(f"{path}: version: must be {TUNING_VERSION}, got {version!r}")

    values: dict[str, object] = {}
    for key, (kind, low, low_in, high, high_in, nullable) in _TUNING_RULES.items():
        if key not in data:
            continue
        value = data[key]
        if value is None:
            if not nullable:
                raise ZoneConfigError(f"{path}: {key}: must not be null")
            values[key] = None
            continue
        if kind == "int":
            if not _is_number(value) or isinstance(value, float):
                raise ZoneConfigError(f"{path}: {key}: must be an integer, got {value!r}")
        number = _number(path, key, value)
        below = number < low if low_in else number <= low
        above = high is not None and (number > high if high_in else number >= high)
        if below or above:
            low_text = f"{'>=' if low_in else '>'} {low:g}"
            high_text = "" if high is None else f" and {'<=' if high_in else '<'} {high:g}"
            null_text = " (or null)" if nullable else ""
            raise ZoneConfigError(f"{path}: {key}: must be {low_text}{high_text}{null_text}, got {value!r}")
        values[key] = int(value) if kind == "int" else number

    tuning = TuningConfig(**values, path=path, sha256=digest)
    if tuning.center_weight + tuning.front_weight <= 0.0:
        raise ZoneConfigError(f"{path}: center_weight, front_weight: must not both be 0")
    warnings = []
    if tuning.release_hold_s < SESSION_CANCEL_CONFIRM_S:
        warnings.append(
            f"{path}: release_hold_s {tuning.release_hold_s:g} is shorter than mirror_session's "
            f"cancel_confirm_s {SESSION_CANCEL_CONFIRM_S:g}: the session may go on to the next person "
            "without returning to STANDBY"
        )
    return tuning, warnings


def tuning_values(tuning: TuningConfig) -> dict[str, object]:
    """The selector parameters only (no provenance), in file order."""
    return {f.name: getattr(tuning, f.name) for f in fields(tuning) if f.name in _TUNING_RULES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m manipulation_execution.actor_zone_config",
        description="Check actor_zone.yaml (and optionally actor_tuning.yaml) the way the node will.",
    )
    parser.add_argument("--zone", required=True, help="actor_zone.yaml")
    parser.add_argument("--tuning", default="", help="actor_tuning.yaml (omit for all defaults)")
    args = parser.parse_args(argv)
    try:
        zone = load_zone(args.zone)
        tuning, warnings = load_tuning(args.tuning)
    except ZoneConfigError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    print("OK")
    print(f"zone {zone.path} sha256={zone.sha256}")
    if tuning.path is None:
        print("tuning (defaults)")
    else:
        print(f"tuning {tuning.path} sha256={tuning.sha256}")
    print("values " + " ".join(f"{k}={v}" for k, v in tuning_values(tuning).items()))
    for warning in warnings:
        print(f"WARN {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFAULT_NEAR_EDGE_MARGIN_PX",
    "MIN_NEAR_EDGE_MARGIN_PX",
    "QUAD_CORNERS",
    "TuningConfig",
    "ZoneConfig",
    "ZoneConfigError",
    "load_tuning",
    "load_zone",
    "tuning_values",
]
