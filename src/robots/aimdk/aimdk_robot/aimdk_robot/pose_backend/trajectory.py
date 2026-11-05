"""Joint-space trajectories through a list of target frames (minimum-jerk, speed-limited)."""

from __future__ import annotations

import numpy as np

from aimdk_robot.pose_backend.model import ARM_JOINTS
from robot_runtime.pose_backend import Trajectory

DEFAULT_SPEED_RADPS = 0.9  # peak; the safety check allows 1.0, the platform presets reach 4.7
MIN_SEGMENT_S = 1.0
RATE_HZ = 50.0


def _min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def segment_duration(a: np.ndarray, b: np.ndarray, speed: float = DEFAULT_SPEED_RADPS) -> float:
    # Peak speed of a minimum-jerk profile is 1.875 x the mean speed.
    return max(MIN_SEGMENT_S, 1.875 * float(np.max(np.abs(b - a))) / speed)


def plan(start_q: dict, frames: list[dict], holds: list[float] | None = None, speed: float = DEFAULT_SPEED_RADPS,
         via: list[list[dict]] | None = None) -> Trajectory:  # fmt: skip
    """Start -> frame 1 (hold) -> frame 2 (hold) ...; `via[k]` are extra waypoints before frame k."""
    holds = holds or [1.0] * len(frames)
    points = [np.array([start_q[n] for n in ARM_JOINTS])]
    times = [0.0]
    t = 0.0
    for k, frame in enumerate(frames):
        targets = [*(via[k] if via else []), frame]
        for target in targets:
            a, b = points[-1], np.array([target[n] for n in ARM_JOINTS])
            dur = segment_duration(a, b, speed)
            n = max(2, int(round(dur * RATE_HZ)))
            for i in range(1, n + 1):
                s = i / n
                points.append(a + (b - a) * _min_jerk(s))
                times.append(t + dur * s)
            t += dur
        hold = float(holds[k])
        if hold > 0:
            points.append(points[-1].copy())
            t += hold
            times.append(t)
    return Trajectory(np.array(times), np.array(points))
