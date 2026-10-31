"""Robot-agnostic plane and rotation geometry for grasp execution.

The generic math (table-plane transformation, axis/quaternion errors and the
rotation helpers) lives here; robot-specific gripper mesh geometry and joint-5
wrist guards live in the robot suites (``so101_suite``) and are loaded only
through configuration-declared providers. This module owns the single
``TablePlane`` type the pick pipeline uses. The rotation helpers came from the
former ``grasp_geometry.py``; its pick-specific candidate scoring lives in
``manipulation_execution.pick.candidate_geometry``.

This module imports nothing from the package.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def euler_xyz_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def quaternion_matrix(quaternion: Quaternion) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_from_matrix(matrix: np.ndarray) -> Quaternion:
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = (
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = (
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = (
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return (
        float(quaternion[0] / norm),
        float(quaternion[1] / norm),
        float(quaternion[2] / norm),
        float(quaternion[3] / norm),
    )


def transform_matrix(translation: Vector3, quaternion: Quaternion) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix(quaternion)
    matrix[:3, 3] = translation
    return matrix


def matrix_from_rowmajor(values: Iterable[float]) -> np.ndarray:
    matrix = np.asarray(list(values), dtype=np.float64).reshape(4, 4)
    matrix[3, :] = [0.0, 0.0, 0.0, 1.0]
    return matrix


@dataclass(frozen=True)
class TablePlane:
    normal: Vector3
    offset: float
    inlier_ratio: float = 0.0


def transform_point(transform: np.ndarray, point: Iterable[float]) -> Vector3:
    value = np.asarray(transform, dtype=np.float64) @ np.array([*point, 1.0], dtype=np.float64)
    return (float(value[0]), float(value[1]), float(value[2]))


def transform_table_plane(
    transform: np.ndarray,
    normal: Iterable[float],
    offset: float,
    *,
    inlier_ratio: float = 0.0,
) -> TablePlane:
    """Transform ``normal dot point + offset = 0`` into the target frame."""

    matrix = np.asarray(transform, dtype=np.float64)
    source_normal = np.asarray(list(normal), dtype=np.float64)
    source_norm = float(np.linalg.norm(source_normal))
    if source_norm <= 1e-9:
        raise ValueError("table plane normal must be non-zero")
    source_normal /= source_norm
    source_offset = float(offset) / source_norm
    target_normal = matrix[:3, :3] @ source_normal
    target_offset = source_offset - float(np.dot(target_normal, matrix[:3, 3]))
    return TablePlane(
        normal=(float(target_normal[0]), float(target_normal[1]), float(target_normal[2])),
        offset=target_offset,
        inlier_ratio=float(inlier_ratio),
    )


def orient_table_plane_upward(plane: TablePlane) -> TablePlane:
    """Canonicalize a table plane so its safe half-space faces base +Z."""

    normal = np.asarray(plane.normal, dtype=np.float64)
    if float(np.dot(normal, (0.0, 0.0, 1.0))) < 0.0:
        return TablePlane(
            normal=tuple(float(value) for value in -normal),
            offset=-float(plane.offset),
            inlier_ratio=float(plane.inlier_ratio),
        )
    return plane


def axis_error_deg(
    planned_quaternion: Quaternion,
    actual_quaternion: Quaternion,
    axis_ee: Iterable[float],
) -> float:
    """Return directed angular error between planned and actual EE axes."""

    axis = np.asarray(list(axis_ee), dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("axis must be non-zero")
    axis /= norm
    planned = quaternion_matrix(planned_quaternion) @ axis
    actual = quaternion_matrix(actual_quaternion) @ axis
    return math.degrees(math.acos(float(np.clip(np.dot(planned, actual), -1.0, 1.0))))


def quaternion_error_deg(planned_quaternion: Quaternion, actual_quaternion: Quaternion) -> float:
    """Return the shortest full-orientation error between two quaternions."""

    planned = np.asarray(planned_quaternion, dtype=np.float64)
    actual = np.asarray(actual_quaternion, dtype=np.float64)
    planned_norm = float(np.linalg.norm(planned))
    actual_norm = float(np.linalg.norm(actual))
    if planned_norm <= 1e-9 or actual_norm <= 1e-9:
        raise ValueError("quaternion must be non-zero")
    dot = abs(float(np.dot(planned / planned_norm, actual / actual_norm)))
    return math.degrees(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))
