"""Pose and joint-path arithmetic shared by every simulation adapter (pure Python, no numpy).

Poses are ``[x, y, z, qw, qx, qy, qz]``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def interpolate(start: Sequence[float], end: Sequence[float], n: int) -> list[list[float]]:
    """``n`` evenly spaced points after ``start``; the last one is exactly ``end``."""
    n = max(1, int(n))
    out = [[a + (b - a) * k / n for a, b in zip(start, end, strict=True)] for k in range(1, n)]
    out.append([float(b) for b in end])
    return out


def nearest_equivalent(start: Sequence[float], target: Sequence[float], limits: Sequence[float]) -> list[float]:
    """Per joint, the ``target + 2*pi*k`` closest to ``start`` that stays within ``|q| <= limit``."""
    out = []
    for i, (a, b) in enumerate(zip(start, target, strict=True)):
        limit = float(limits[i]) if i < len(limits) else math.pi
        candidates = [b + 2.0 * math.pi * k for k in (-1, 0, 1)]
        candidates = [c for c in candidates if abs(c) <= limit] or [b]
        out.append(min(candidates, key=lambda c: abs(c - a)))
    return out


def rotate(q: Sequence[float], v: Sequence[float]) -> list[float]:
    """Rotate vector ``v`` by the ``[qw, qx, qy, qz]`` quaternion ``q``."""
    w, x, y, z = (float(c) for c in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    vx, vy, vz = (float(c) for c in v)
    # t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)]


def shift_pose(pose: Sequence[float], offset: Sequence[float], sign: float = 1.0) -> list[float]:
    """``pose`` ([x, y, z, qw, qx, qy, qz]) moved by ``sign * offset`` expressed in its own frame."""
    pose = [float(v) for v in pose]
    if not any(offset) or len(pose) < 7:
        return pose
    d = rotate(pose[3:7], offset)
    return [pose[0] + sign * d[0], pose[1] + sign * d[1], pose[2] + sign * d[2], *pose[3:7]]


def pitch_pose(pose: Sequence[float], degrees: float) -> list[float]:
    """``pose`` with its orientation turned by ``degrees`` about its own y axis (same position)."""
    pose = [float(v) for v in pose]
    half = math.radians(degrees) / 2.0
    w1, x1, y1, z1 = pose[3:7]
    w2, y2 = math.cos(half), math.sin(half)  # q2 = (w2, 0, y2, 0)
    q = [w1 * w2 - y1 * y2, x1 * w2 - z1 * y2, w1 * y2 + y1 * w2, z1 * w2 + x1 * y2]
    return [*pose[:3], *q]


def position_error(target_pose: Sequence[float], ee_pose: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(target_pose[:3], ee_pose[:3], strict=False)))


def orientation_error(target_pose: Sequence[float], ee_pose: Sequence[float]) -> float:
    """Angle between two ``[qw, qx, qy, qz]`` quaternions (entries 3..6 of a pose)."""
    a, b = target_pose[3:7], ee_pose[3:7]
    if len(a) < 4 or len(b) < 4:
        return 0.0
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    dot = abs(sum(x * y for x, y in zip(a, b, strict=True))) / (na * nb)
    return 2.0 * math.acos(min(1.0, dot))
