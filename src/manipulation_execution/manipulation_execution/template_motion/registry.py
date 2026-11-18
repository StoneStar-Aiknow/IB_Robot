"""The segment whitelist (``segments.json``, schema ``x2_template_segments/v1``).

The same file the data plane's playback port reads once at startup
(``template-play-M0-interface.md`` §1). Each segment is checked on its own; a
segment that fails is left out with its reason and the rest are used. The
data plane checks again with its own limits and has the last word: a
segment it refused never reaches ``ENTER_B`` (its status lists the loaded
segments, and the blender only starts those).

Checks here, all the data plane's plus one of the session's own:

* 14 arm and 20 hand values, all finite; ``t_in``, ``t_out`` in [0.3, 2.0],
  ``t_out_stop`` in [0.3, 0.6] (M0 revision 2), ``hold_max_s`` in (0, 10], both tolerances
  finite and positive, ``report_sha256`` 64 hex digits (format only);
* speed: for each of ``t_in``, ``t_out``, ``t_out_stop`` and every joint,
  1.875 * |T - P| / t <= ``velocity`` (2 rad/s, the CBF's 0.04 per 20 ms);
* arm limits, when given (the CBF's lower/upper, already shrunk by 1 degree);
* hands inside ``HandOverlay``'s ranges (session only): the overlay clips its
  output to them, so a hand_P outside could never be reached and blend A
  would always time out.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from manipulation_execution.mirror_session.endpoints import ARM_JOINT_COUNT, HAND_JOINT_COUNT

SCHEMA = "x2_template_segments/v1"
VELOCITY_RAD_S = 2.0
MIN_JERK_PEAK = 1.875  # peak of d/ds (10 s^3 - 15 s^4 + 6 s^5)
_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Segment:
    name: str
    P: tuple[float, ...]
    T: tuple[float, ...]
    hand_P: tuple[float, ...]
    hand_T: tuple[float, ...]
    t_in: float
    t_out: float
    t_out_stop: float
    hold_max_s: float
    start_tol_rad: float
    hand_start_tol: float
    report_sha256: str


def _vector(raw: Mapping, key: str, length: int) -> tuple[float, ...]:
    values = raw.get(key)
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{key} must list {length} numbers")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError(f"{key} must be finite numbers")
    return tuple(float(v) for v in values)


def _number(raw: Mapping, key: str, low: float, high: float, low_open: bool) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    value = float(value)
    if (value <= low if low_open else value < low) or value > high:
        raise ValueError(f"{key}={value:g} is outside {'(' if low_open else '['}{low:g}, {high:g}]")
    return value


def check_segment(
    name: str,
    raw: object,
    arm_limits: Sequence[tuple[float, float]] | None = None,
    velocity: float = VELOCITY_RAD_S,
    hand_limits: tuple[Sequence[float], Sequence[float]] | None = None,
) -> Segment:
    """One whitelist entry, or ValueError naming the field, the value and the bound."""
    if not isinstance(raw, Mapping):
        raise ValueError("entry must be a mapping")
    P, T = _vector(raw, "P", ARM_JOINT_COUNT), _vector(raw, "T", ARM_JOINT_COUNT)
    hand_P, hand_T = _vector(raw, "hand_P", HAND_JOINT_COUNT), _vector(raw, "hand_T", HAND_JOINT_COUNT)
    t_in = _number(raw, "t_in", 0.3, 2.0, False)
    t_out = _number(raw, "t_out", 0.3, 2.0, False)
    t_out_stop = _number(raw, "t_out_stop", 0.3, 0.6, False)
    hold_max_s = _number(raw, "hold_max_s", 0.0, 10.0, True)
    start_tol = _number(raw, "start_tol_rad", 0.0, math.inf, True)
    hand_tol = _number(raw, "hand_start_tol", 0.0, math.inf, True)
    sha = raw.get("report_sha256")
    if not isinstance(sha, str) or not _SHA.match(sha):
        raise ValueError("report_sha256 must be 64 lowercase hex digits")
    for duration_name, duration in (("t_in", t_in), ("t_out", t_out), ("t_out_stop", t_out_stop)):
        for j, (p, t) in enumerate(zip(P, T, strict=False)):
            speed = MIN_JERK_PEAK * abs(t - p) / duration
            if not speed <= velocity:
                raise ValueError(
                    f"joint {j}: 1.875*|T-P|/{duration_name} = {speed:.3f} rad/s exceeds {velocity:g} rad/s"
                )
    if arm_limits is not None:
        for pose_name, pose in (("P", P), ("T", T)):
            for j, (value, (low, high)) in enumerate(zip(pose, arm_limits, strict=False)):
                if not low <= value <= high:
                    raise ValueError(f"{pose_name}[{j}]={value:.4f} is outside [{low:.4f}, {high:.4f}]")
    if hand_limits is not None:
        low, high = hand_limits
        for pose_name, pose in (("hand_P", hand_P), ("hand_T", hand_T)):
            for j, value in enumerate(pose):
                if not low[j] - 1e-9 <= value <= high[j] + 1e-9:
                    raise ValueError(
                        f"{pose_name}[{j}]={value:.4f} is outside the hand range [{low[j]:.4f}, {high[j]:.4f}]"
                    )
    return Segment(name, P, T, hand_P, hand_T, t_in, t_out, t_out_stop, hold_max_s, start_tol, hand_tol, sha)


def parse_segments(
    content: object,
    arm_limits: Sequence[tuple[float, float]] | None = None,
    velocity: float = VELOCITY_RAD_S,
    hand_limits: tuple[Sequence[float], Sequence[float]] | None = None,
) -> tuple[dict[str, Segment], dict[str, str]]:
    """(segments that passed, {name: reason} for those that did not). ValueError if the file is unusable."""
    if not isinstance(content, Mapping) or content.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} document")
    entries = content.get("segments")
    if not isinstance(entries, Mapping):
        raise ValueError("segments must be a mapping")
    loaded, rejected = {}, {}
    for name, raw in entries.items():
        try:
            if not isinstance(name, str) or not name:
                raise ValueError("segment name must be a non-empty string")
            loaded[name] = check_segment(name, raw, arm_limits, velocity, hand_limits)
        except ValueError as exc:
            rejected[str(name)] = str(exc)
    return loaded, rejected


def load_segments(path: str, **checks) -> tuple[dict[str, Segment], dict[str, str]]:
    with open(path, encoding="utf-8") as stream:
        content = json.load(stream)
    return parse_segments(content, **checks)
