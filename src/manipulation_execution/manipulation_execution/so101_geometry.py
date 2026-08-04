"""SO101 target-gripper geometry checks migrated from the proven pick script."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import ConvexHull, QhullError
except ImportError:  # pragma: no cover - full vertices remain geometrically correct.
    ConvexHull = None
    QhullError = ValueError

from manipulation_execution.grasp_geometry import euler_xyz_matrix, quaternion_matrix

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


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


def _matrix_from_xyz_rpy(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = euler_xyz_matrix(rpy)
    matrix[:3, 3] = [float(value) for value in xyz]
    return matrix


@lru_cache(maxsize=4)
def _read_stl_vertices(path: str, max_triangles: int = 2500) -> np.ndarray:
    mesh_path = Path(path)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    data = mesh_path.read_bytes()
    triangles = np.zeros((0, 3, 3), dtype=np.float64)
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if triangle_count > 0 and 84 + triangle_count * 50 == len(data):
            dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
            raw = np.frombuffer(data, dtype=dtype, count=triangle_count, offset=84)
            triangles = raw["vertices"].astype(np.float64)
    if len(triangles) == 0:
        vertices = []
        for line in data.decode("ascii", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] == "vertex":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(vertices) >= 3:
            triangles = np.asarray(vertices[: len(vertices) // 3 * 3], dtype=np.float64).reshape(-1, 3, 3)
    if len(triangles) == 0:
        raise ValueError(f"No STL triangles found in {mesh_path}")
    step = max(1, int(math.ceil(len(triangles) / max_triangles)))
    return triangles[::step].reshape(-1, 3)


@lru_cache(maxsize=4)
def _read_stl_convex_hull_vertices(path: str, max_triangles: int = 2500) -> np.ndarray:
    vertices = np.unique(_read_stl_vertices(path, max_triangles), axis=0)
    if ConvexHull is None or len(vertices) < 4:
        return vertices
    try:
        return vertices[ConvexHull(vertices).vertices]
    except QhullError:
        return vertices


def _width_to_jaw_angle(width_m: float | None) -> float:
    if width_m is None or not math.isfinite(float(width_m)):
        return 0.45
    normalized = (float(width_m) - 0.008) / (0.080 - 0.008)
    return max(0.0, min(1.0, normalized))


@lru_cache(maxsize=4)
def _gripper_geometry_data(mesh_directory: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directory = Path(mesh_directory)
    fixed_vertices = _read_stl_convex_hull_vertices(str(directory / "wrist_roll_follower_so101_v1.stl"))
    moving_vertices = _read_stl_convex_hull_vertices(str(directory / "moving_jaw_so101_v1.stl"))
    gripper_visual = _matrix_from_xyz_rpy(
        (5.55112e-17, -0.000218214, 0.000949706),
        (-3.14159, -5.55112e-17, -9.17912e-24),
    )
    gripper_to_jaw = _matrix_from_xyz_rpy((0.0202, 0.0188, -0.0234), (1.5708, 0.209440, 0.000001))
    jaw_visual = _matrix_from_xyz_rpy((-5.55112e-17, -1.94746e-17, 0.0189), (9.53145e-17, -4.66093e-24, 0.0))
    return fixed_vertices, moving_vertices, gripper_visual, gripper_to_jaw, jaw_visual


def gripper_mesh_vertices(
    mesh_directory: Path,
    xyz: Vector3,
    quaternion: Quaternion,
    width_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed and moving SO101 mesh vertices in the base frame."""

    base_to_gripper = np.eye(4, dtype=np.float64)
    base_to_gripper[:3, :3] = quaternion_matrix(quaternion)
    base_to_gripper[:3, 3] = xyz
    fixed_local, jaw_local, gripper_visual, gripper_to_jaw, jaw_visual = _gripper_geometry_data(str(mesh_directory))
    jaw_motion = _matrix_from_xyz_rpy((0.0, 0.0, 0.0), (0.0, 0.0, _width_to_jaw_angle(width_m)))

    def apply(transform: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        vertices_h = np.hstack([vertices, np.ones((len(vertices), 1), dtype=np.float64)])
        return (transform[:3, :] @ vertices_h.T).T

    return (
        apply(base_to_gripper @ gripper_visual, fixed_local),
        apply(base_to_gripper @ gripper_to_jaw @ jaw_motion @ jaw_visual, jaw_local),
    )


def gripper_mesh_min_z(
    mesh_directory: Path,
    xyz: Vector3,
    quaternion: Quaternion,
    width_m: float | None,
) -> float:
    meshes = gripper_mesh_vertices(mesh_directory, xyz, quaternion, width_m)
    return min(float(vertices[:, 2].min()) for vertices in meshes if len(vertices))


def tabletop_clearance(
    mesh_directory: Path,
    approach: Vector3,
    grasp: Vector3,
    quaternion: Quaternion,
    width_m: float | None,
    plane: TablePlane,
    *,
    sweep_steps: int,
) -> float:
    del sweep_steps
    normal = np.asarray(plane.normal, dtype=np.float64)
    grasp_clearance = math.inf
    for vertices in gripper_mesh_vertices(mesh_directory, grasp, quaternion, width_m):
        if len(vertices):
            grasp_clearance = min(grasp_clearance, float(np.min(vertices @ normal + plane.offset)))
    if not math.isfinite(grasp_clearance):
        raise ValueError("SO101 gripper meshes contain no vertices")
    translation = np.asarray(approach, dtype=np.float64) - np.asarray(grasp, dtype=np.float64)
    approach_clearance = grasp_clearance + float(translation @ normal)
    return min(grasp_clearance, approach_clearance)


def gripper_geometry_metrics_batch(
    mesh_directory: Path,
    candidates: Sequence[tuple[Vector3, Vector3, Quaternion, float | None]],
    plane: TablePlane | None,
    *,
    clearance_threshold_m: float = 0.0,
    threshold_fallback_m: float = 1e-5,
) -> list[tuple[float, float | None]]:
    """Vectorize SO101 mesh height and tabletop checks across candidates."""

    if not candidates:
        return []
    fixed_local, moving_local, gripper_visual, gripper_to_jaw, jaw_visual = _gripper_geometry_data(str(mesh_directory))
    fixed_transforms = []
    moving_transforms = []
    for _, grasp, quaternion, width_m in candidates:
        base_to_gripper = np.eye(4, dtype=np.float64)
        base_to_gripper[:3, :3] = quaternion_matrix(quaternion)
        base_to_gripper[:3, 3] = grasp
        jaw_motion = _matrix_from_xyz_rpy((0.0, 0.0, 0.0), (0.0, 0.0, _width_to_jaw_angle(width_m)))
        fixed_transforms.append(base_to_gripper @ gripper_visual)
        moving_transforms.append(base_to_gripper @ gripper_to_jaw @ jaw_motion @ jaw_visual)

    fixed_transform_array = np.stack(fixed_transforms)
    moving_transform_array = np.stack(moving_transforms)
    fixed_world = (
        np.matmul(fixed_local[None, :, :], np.swapaxes(fixed_transform_array[:, :3, :3], 1, 2))
        + fixed_transform_array[:, None, :3, 3]
    )
    moving_world = (
        np.matmul(moving_local[None, :, :], np.swapaxes(moving_transform_array[:, :3, :3], 1, 2))
        + moving_transform_array[:, None, :3, 3]
    )
    minimum_z = np.minimum(np.min(fixed_world[:, :, 2], axis=1), np.min(moving_world[:, :, 2], axis=1))
    if plane is None:
        return [(float(value), None) for value in minimum_z]

    normal = np.asarray(plane.normal, dtype=np.float64)
    fixed_grasp = np.min(fixed_world @ normal + plane.offset, axis=1)
    moving_grasp = np.min(moving_world @ normal + plane.offset, axis=1)
    grasp_clearance = np.minimum(fixed_grasp, moving_grasp)
    approaches = np.asarray([candidate[0] for candidate in candidates], dtype=np.float64)
    grasps = np.asarray([candidate[1] for candidate in candidates], dtype=np.float64)
    approach_clearance = grasp_clearance + (approaches - grasps) @ normal
    clearances = np.minimum(grasp_clearance, approach_clearance)

    near_threshold = np.abs(clearances - float(clearance_threshold_m)) <= max(0.0, float(threshold_fallback_m))
    for index in np.flatnonzero(near_threshold):
        approach, grasp, quaternion, width_m = candidates[int(index)]
        minimum_z[index] = gripper_mesh_min_z(mesh_directory, grasp, quaternion, width_m)
        clearances[index] = tabletop_clearance(
            mesh_directory,
            approach,
            grasp,
            quaternion,
            width_m,
            plane,
            sweep_steps=1,
        )
    return [(float(z_value), float(clearance)) for z_value, clearance in zip(minimum_z, clearances, strict=True)]
