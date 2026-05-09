"""Probe a V4L2 device for exposure-control unit and current frame rate.

Two pieces of information the 4-stage hardware calibrator needs that
``v4l2_ctl.py`` does not yet expose:

1. **Exposure unit** — UVC ``exposure_time_absolute`` /
   ``exposure_absolute`` are 100 us per LSB; non-UVC drivers may expose a
   plain ``exposure`` integer with no documented unit. We need to know
   which one to convert ``--max-exposure-ms`` into raw ticks.

2. **Current frame rate** — to derive ``exp_max_us = min(cli_max,
   1e6/fps - 2000)`` so the calibrator never picks an exposure value
   that would cause a dropped frame at the active streaming fps.

This module is import-light and never raises on probe failure; callers
get a structured fallback so the calibrator can degrade gracefully when
``v4l2-ctl`` is missing or the device is busy.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


# Ordered new-name first; first match wins.
_EXPOSURE_ABSOLUTE_NAMES: tuple[str, ...] = (
    "exposure_time_absolute",
    "exposure_absolute",
)
_EXPOSURE_PLAIN_NAMES: tuple[str, ...] = (
    "exposure",
    "exposure_time",
)
_UVC_ABSOLUTE_UNIT_US = 100


@dataclass(frozen=True)
class ExposureProbe:
    """Probed exposure-control characteristics for a V4L2 device.

    Attributes
    ----------
    control_name:
        The actual V4L2 control name found on the device, or empty string
        when none of the known names matched.
    unit_us:
        Microseconds per LSB. ``100`` for UVC absolute controls;
        ``None`` when the unit is unknown (caller must treat the value
        as a raw integer and not display millisecond hints).
    fps:
        Streaming frame rate (frames per second) reported by
        ``v4l2-ctl --get-parm``. ``None`` when not detected — caller
        should default to 30 fps per project plan §8.1.
    """

    control_name: str
    unit_us: int | None
    fps: float | None


def _run(args: list[str], timeout: float = 2.0) -> tuple[int, str]:
    """Run a v4l2-ctl invocation. Returns (rc, stdout). Never raises."""
    try:
        cp = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
        )
        return cp.returncode, cp.stdout or ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return -1, ""


def _detect_control_name(device: str) -> str:
    """Return the first matching exposure control name, or '' if none."""
    rc, out = _run(["v4l2-ctl", "--device", device, "--list-ctrls"])
    if rc != 0:
        return ""
    available = set()
    for line in out.splitlines():
        s = line.strip()
        if " : " not in s:
            continue
        head = s.split(" : ", 1)[0]
        toks = head.split()
        if toks:
            available.add(toks[0])
    for name in _EXPOSURE_ABSOLUTE_NAMES:
        if name in available:
            return name
    for name in _EXPOSURE_PLAIN_NAMES:
        if name in available:
            return name
    return ""


def _unit_us_for(control_name: str) -> int | None:
    if control_name in _EXPOSURE_ABSOLUTE_NAMES:
        return _UVC_ABSOLUTE_UNIT_US
    return None


# Example output of `v4l2-ctl --get-parm`:
#   Streaming Parameters Video Capture:
#       Capabilities     : timeperframe
#       Frames per second: 30.000 (30/1)
_FPS_PATTERN = re.compile(r"Frames per second:\s*([0-9.]+)")


def _detect_fps(device: str) -> float | None:
    rc, out = _run(["v4l2-ctl", "--device", device, "--get-parm"])
    if rc != 0:
        return None
    m = _FPS_PATTERN.search(out)
    if not m:
        return None
    try:
        fps = float(m.group(1))
    except ValueError:
        return None
    if fps <= 0.0:
        return None
    return fps


def probe_exposure(device: str) -> ExposureProbe:
    """Probe *device* for exposure control name + unit + fps.

    Always returns; never raises. On full failure all fields are empty /
    None and the caller falls back to project defaults.
    """
    control_name = _detect_control_name(device)
    unit_us = _unit_us_for(control_name) if control_name else None
    fps = _detect_fps(device)
    return ExposureProbe(
        control_name=control_name,
        unit_us=unit_us,
        fps=fps,
    )


# --- Public derived helpers -------------------------------------------------

DEFAULT_FPS_FALLBACK = 30.0
DEFAULT_MAX_EXPOSURE_MS = 15.0
_FRAME_TIME_HEADROOM_US = 2000
#: Above this fps, we treat the v4l2-ctl `--get-parm` report as
#: unreliable and fall back to ``DEFAULT_FPS_FALLBACK`` for the
#: frame-time safety calc. Rationale: consumer UVC cameras advertise
#: the *fastest available* mode in `--get-parm` when no application
#: has called VIDIOC_S_PARM, but their actual streaming rate during
#: calibration is almost always ≤60 fps. A bogus high fps crushes
#: ``exp_max`` to a few ticks (e.g. fps=400 → 500 µs = 5 ticks at
#: 100 µs/LSB) and makes the exposure stage degenerate. Set to
#: ``float("inf")`` to disable the clamp.
_FPS_PROBE_SANITY_MAX = 120.0


def _sanitize_fps(fps: float | None) -> float:
    """Return a trustworthy fps for the frame-time safety calc.

    Returns ``DEFAULT_FPS_FALLBACK`` when *fps* is unknown, non-positive,
    or above ``_FPS_PROBE_SANITY_MAX`` (likely a v4l2 misreport).
    """
    if fps is None or fps <= 0.0 or fps > _FPS_PROBE_SANITY_MAX:
        return DEFAULT_FPS_FALLBACK
    return float(fps)


def compute_exposure_max_us(
    cli_max_ms: float | None,
    fps: float | None,
) -> int:
    """Return ``exp_max_us`` per plan §2 Stage 1.

    ``min(cli_max_ms*1000, 1e6/fps - headroom)`` with sane fallbacks:
        - cli_max_ms None ⇒ DEFAULT_MAX_EXPOSURE_MS (15 ms).
        - fps None / <=0 / unreliably high (>120 fps) ⇒
          DEFAULT_FPS_FALLBACK (30 fps). See ``_sanitize_fps``.
    """
    cli_us = float(
        DEFAULT_MAX_EXPOSURE_MS if cli_max_ms is None else cli_max_ms
    ) * 1000.0
    eff_fps = _sanitize_fps(fps)
    fps_us = max(1.0, 1_000_000.0 / eff_fps - _FRAME_TIME_HEADROOM_US)
    return int(min(cli_us, fps_us))


def ticks_from_us(us: float, unit_us: int | None) -> int:
    """Convert a microsecond budget to raw V4L2 control ticks.

    When the unit is unknown the caller is expected to have given a raw
    integer through a different code path; we still return ``int(us)``
    so the function is never lossy by accident — callers must check
    ``unit_us`` themselves to decide whether the interpretation is
    physically meaningful.
    """
    if unit_us is None or unit_us <= 0:
        return int(us)
    return max(1, int(round(us / unit_us)))
