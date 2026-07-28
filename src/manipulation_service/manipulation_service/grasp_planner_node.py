"""GraspPlanner ROS 2 node integrating GraspGen with the detection pipeline."""

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from ibrobot_msgs.msg import GraspCandidate, GraspCandidateArray
from ibrobot_msgs.srv import DetectSegment, PlanGrasp

from .graspgen_wrapper import (
    DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP,
    GraspDiagnostic,
    GraspGenWrapper,
    TablePlaneFit,
    fit_execution_table_plane,
)

_DEBUG_OUTPUT_DEFAULT = "default"
_DEBUG_OUTPUT_NONE = "none"
_DEBUG_OUTPUT_DIAGNOSTIC = "diagnostic"
_DEBUG_OUTPUT_FULL = "full"
_VALID_DEBUG_OUTPUT_MODES = {
    "",
    _DEBUG_OUTPUT_DEFAULT,
    _DEBUG_OUTPUT_NONE,
    _DEBUG_OUTPUT_DIAGNOSTIC,
    _DEBUG_OUTPUT_FULL,
}


@dataclass
class DepthFrame:
    stamp_ns: int
    frame_id: str
    data: np.ndarray
    depth_scale: float


@dataclass
class CameraInfoFrame:
    stamp_ns: int
    msg: CameraInfo


@dataclass
class DebugPreviewCandidate:
    pose_4x4: np.ndarray
    confidence: float
    collision_free: bool


def _stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _stamp_from_ns(stamp_ns: int) -> TimeMsg:
    stamp_ns = max(0, int(stamp_ns))
    return TimeMsg(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)


def _depth_scale_for_msg(msg: Image, fallback_scale: float) -> float:
    if msg.encoding in ("32FC1", "64FC1"):
        return 1.0
    return fallback_scale


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "prompt"


def _parse_float_csv(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def _parse_vector3_parameter(value: str, name: str) -> list[float]:
    values = _parse_float_csv(value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated floats")
    return values


def _diagnostic_to_lines(diag: GraspDiagnostic, *, detection_confidence: float | None = None) -> list[str]:
    lines = []
    if diag.failure_stage:
        lines.extend([f"failure_stage: {diag.failure_stage}", f"failure_reason: {diag.failure_reason}"])
    lines.append("detection_status: ok")
    if detection_confidence is not None:
        lines.append(f"detection_confidence: {detection_confidence:.4f}")
    lines.extend(
        [
            f"mask_shape: {diag.mask_shape}",
            f"depth_shape: {diag.depth_shape}",
            f"mask_resized_to_depth: {diag.mask_resized}",
            f"mask_pixel_count: {diag.mask_pixel_count}",
            f"valid_depth_pixel_count: {diag.valid_depth_pixel_count}",
            f"valid_depth_in_mask_count: {diag.valid_depth_in_mask_count}",
            f"valid_depth_ratio_in_mask: {diag.valid_depth_ratio_in_mask:.3f}",
            f"object_cloud_completion_enabled: {diag.object_cloud_completion_enabled}",
            f"object_cloud_completion_mode: {diag.object_cloud_completion_mode}",
            f"object_point_count_raw: {diag.object_point_count_raw}",
            f"object_point_count_completed: {diag.object_point_count_completed}",
            f"object_completion_added_count: {diag.object_completion_added_count}",
            f"object_prismatic_extrude_enabled: {diag.object_prismatic_extrude_enabled}",
            f"object_prismatic_extrude_added_count: {diag.object_prismatic_extrude_added_count}",
            f"object_point_count_graspgen_input: {diag.object_point_count_graspgen_input}",
            f"scene_cloud_table_holes_enabled: {diag.scene_cloud_table_holes_enabled}",
            f"scene_point_count_raw: {diag.scene_point_count_raw}",
            f"scene_table_hole_added_count: {diag.scene_table_hole_added_count}",
            f"object_point_count: {diag.object_point_count}",
            f"scene_point_count: {diag.scene_point_count}",
            f"raw_grasp_count: {diag.raw_grasp_count}",
            f"collision_filter: {diag.collision_filter_after}/{diag.collision_filter_before}",
            f"tabletop_plane_found: {diag.tabletop_plane_found}",
            "tabletop_plane_normal: "
            + (
                "unavailable"
                if diag.tabletop_plane_normal is None
                else ",".join(f"{float(value):.6f}" for value in diag.tabletop_plane_normal)
            ),
            f"tabletop_plane_d: {diag.tabletop_plane_d:.6f}",
            f"tabletop_inlier_ratio: {diag.tabletop_inlier_ratio:.3f}",
            f"tabletop_best_inlier_ratio: {diag.tabletop_best_inlier_ratio:.3f}",
            f"tabletop_filter_mode: {diag.tabletop_filter_mode}",
            f"source_gripper_tabletop_sweep_enabled: {diag.source_gripper_tabletop_sweep_enabled}",
            f"tabletop_low_profile: {diag.tabletop_low_profile}",
            f"tabletop_relaxed: {diag.tabletop_relaxed}",
            f"tabletop_object_height_m: {diag.tabletop_object_height_m:.4f}",
            f"tabletop_clearance_used_m: {diag.tabletop_clearance_used_m:.4f}",
            f"tabletop_pregrasp_distance_used_m: {diag.tabletop_pregrasp_distance_used_m:.4f}",
            f"tabletop_best_candidate_clearance_m: {diag.tabletop_best_candidate_clearance_m:.4f}",
            f"tabletop_auto_tuned: {diag.tabletop_auto_tuned}",
            f"tabletop_auto_tune_attempts: {diag.tabletop_auto_tune_attempts}",
            f"tabletop_auto_tune_reason: {diag.tabletop_auto_tune_reason}",
            f"tabletop_filter: {diag.tabletop_filter_after}/{diag.tabletop_filter_before}",
        ]
    )
    if diag.tabletop_failure_reason:
        lines.append(f"tabletop_failure_reason: {diag.tabletop_failure_reason}")
    return lines


def _diagnostic_to_dict(diag: GraspDiagnostic, *, detection_confidence: float | None = None) -> dict:
    data = {
        "failure_stage": diag.failure_stage,
        "failure_reason": diag.failure_reason,
        "mask_shape": diag.mask_shape,
        "depth_shape": diag.depth_shape,
        "mask_resized_to_depth": bool(diag.mask_resized),
        "mask_pixel_count": int(diag.mask_pixel_count),
        "valid_depth_pixel_count": int(diag.valid_depth_pixel_count),
        "valid_depth_in_mask_count": int(diag.valid_depth_in_mask_count),
        "valid_depth_ratio_in_mask": float(diag.valid_depth_ratio_in_mask),
        "object_cloud_completion_enabled": bool(diag.object_cloud_completion_enabled),
        "object_cloud_completion_mode": diag.object_cloud_completion_mode,
        "object_point_count_raw": int(diag.object_point_count_raw),
        "object_point_count_completed": int(diag.object_point_count_completed),
        "object_completion_added_count": int(diag.object_completion_added_count),
        "object_prismatic_extrude_enabled": bool(diag.object_prismatic_extrude_enabled),
        "object_prismatic_extrude_added_count": int(diag.object_prismatic_extrude_added_count),
        "object_point_count_graspgen_input": int(diag.object_point_count_graspgen_input),
        "scene_cloud_table_holes_enabled": bool(diag.scene_cloud_table_holes_enabled),
        "scene_point_count_raw": int(diag.scene_point_count_raw),
        "scene_table_hole_added_count": int(diag.scene_table_hole_added_count),
        "object_point_count": int(diag.object_point_count),
        "scene_point_count": int(diag.scene_point_count),
        "raw_grasp_count": int(diag.raw_grasp_count),
        "collision_filter_before": int(diag.collision_filter_before),
        "collision_filter_after": int(diag.collision_filter_after),
        "tabletop_filter_before": int(diag.tabletop_filter_before),
        "tabletop_filter_after": int(diag.tabletop_filter_after),
        "tabletop_plane_found": bool(diag.tabletop_plane_found),
        "tabletop_plane_normal": (
            None if diag.tabletop_plane_normal is None else [float(value) for value in diag.tabletop_plane_normal]
        ),
        "tabletop_plane_d": float(diag.tabletop_plane_d),
        "tabletop_object_top_xyz": (
            None if diag.tabletop_object_top_xyz is None else [float(value) for value in diag.tabletop_object_top_xyz]
        ),
        "tabletop_inlier_ratio": float(diag.tabletop_inlier_ratio),
        "tabletop_best_inlier_ratio": float(diag.tabletop_best_inlier_ratio),
        "tabletop_failure_reason": diag.tabletop_failure_reason,
        "tabletop_filter_mode": diag.tabletop_filter_mode,
        "source_gripper_tabletop_sweep_enabled": bool(diag.source_gripper_tabletop_sweep_enabled),
        "tabletop_low_profile": bool(diag.tabletop_low_profile),
        "tabletop_relaxed": bool(diag.tabletop_relaxed),
        "tabletop_object_height_m": float(diag.tabletop_object_height_m),
        "tabletop_object_height_p05_m": float(diag.tabletop_object_height_p05_m),
        "tabletop_object_height_median_m": float(diag.tabletop_object_height_median_m),
        "tabletop_object_height_p95_m": float(diag.tabletop_object_height_p95_m),
        "tabletop_clearance_used_m": float(diag.tabletop_clearance_used_m),
        "tabletop_pregrasp_distance_used_m": float(diag.tabletop_pregrasp_distance_used_m),
        "tabletop_best_candidate_clearance_m": float(diag.tabletop_best_candidate_clearance_m),
        "tabletop_worst_candidate_clearance_m": float(diag.tabletop_worst_candidate_clearance_m),
        "tabletop_auto_tuned": bool(diag.tabletop_auto_tuned),
        "tabletop_auto_tune_attempts": int(diag.tabletop_auto_tune_attempts),
        "tabletop_auto_tune_reason": diag.tabletop_auto_tune_reason,
    }
    if detection_confidence is not None:
        data["detection_confidence"] = float(detection_confidence)
    return data


def _normalize_debug_output_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in _VALID_DEBUG_OUTPUT_MODES:
        valid = ", ".join(sorted(m or "empty" for m in _VALID_DEBUG_OUTPUT_MODES))
        raise ValueError(f"invalid debug_output_mode {value!r}; expected one of: {valid}")
    return mode or _DEBUG_OUTPUT_DEFAULT


def _debug_output_enabled(mode: str) -> bool:
    return mode in {_DEBUG_OUTPUT_DIAGNOSTIC, _DEBUG_OUTPUT_FULL}


def _candidate_debug_record(index: int, candidate) -> dict:
    return {
        "index": index,
        "confidence": round(float(candidate.confidence), 6),
        "collision_free": bool(candidate.collision_free),
        "target_width_m": round(float(candidate.target_width_m), 6),
        "target_width_quality": round(float(candidate.target_width_quality), 3),
        "width_axis_camera": [round(float(v), 6) for v in candidate.width_axis_camera],
        "target_width_min_offset_m": round(float(candidate.target_width_min_offset_m), 6),
        "target_width_max_offset_m": round(float(candidate.target_width_max_offset_m), 6),
        "position_xyz": [round(float(v), 6) for v in candidate.pose_4x4[:3, 3]],
        "pose_4x4_rowmajor": candidate.pose_4x4.flatten().tolist(),
    }


def _debug_points_or_fallback(points: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    if points is None:
        return fallback.astype(np.float32, copy=False)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return fallback.astype(np.float32, copy=False)
    return arr


def _sample_execution_table_points(points: np.ndarray | None, max_points: int) -> np.ndarray:
    if points is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    step = max(1, len(arr) // max(1, int(max_points)))
    return arr[::step]


def _fit_execution_table_plane(diagnostic: GraspDiagnostic) -> TablePlaneFit:
    thread = getattr(diagnostic, "_execution_table_fit_thread", None)
    holder = getattr(diagnostic, "_execution_table_fit_holder", None)
    if thread is not None and isinstance(holder, dict):
        thread.join()
        error = holder.get("error")
        if isinstance(error, Exception):
            raise error
        fit = holder.get("fit")
        if isinstance(fit, TablePlaneFit):
            return fit

    object_source = diagnostic.object_pc_after_completion
    if object_source is None:
        object_source = diagnostic.object_pc_raw
    return fit_execution_table_plane(diagnostic.scene_pc_after_completion, object_source)


def _object_stage_colors(total: int, raw_count: int, inpaint_count: int) -> np.ndarray:
    colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (max(0, int(total)), 1))
    raw_end = min(max(0, int(raw_count)), len(colors))
    inpaint_end = min(raw_end + max(0, int(inpaint_count)), len(colors))
    if inpaint_end > raw_end:
        colors[raw_end:inpaint_end] = np.array([255, 170, 0], dtype=np.uint8)
    if inpaint_end < len(colors):
        colors[inpaint_end:] = np.array([0, 200, 255], dtype=np.uint8)
    return colors


def _scene_stage_colors(total: int, raw_count: int) -> np.ndarray:
    colors = np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (max(0, int(total)), 1))
    raw_end = min(max(0, int(raw_count)), len(colors))
    if raw_end < len(colors):
        colors[raw_end:] = np.array([168, 85, 247], dtype=np.uint8)
    return colors


def _save_ply(pts, colors, out_path: Path):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(out_path), pcd)


def _confidence_color(confidence):
    t = float(np.clip(confidence, 0.0, 1.0))
    return [1.0 - t, t, 0.0]


def _rgb_string(color, alpha=None):
    rgb = [int(round(float(c) * 255.0)) for c in color]
    if alpha is None:
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


def _make_gripper_mesh_visual(gripper_mesh_template, transform, color):
    import open3d as o3d

    mesh = gripper_mesh_template.copy()
    mesh.apply_transform(transform)
    mesh_visual = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    mesh_visual.compute_vertex_normals()
    mesh_visual.paint_uniform_color(color)
    return mesh_visual


def _save_gripper_meshes(candidates, gripper_name, out_path: Path, max_grasps: int):
    if not candidates or max_grasps <= 0:
        return

    import open3d as o3d
    from grasp_gen.robot import get_gripper_info

    gripper_info = get_gripper_info(gripper_name)
    gripper_mesh_template = gripper_info.collision_mesh
    combined = o3d.geometry.TriangleMesh()

    for g in candidates[:max_grasps]:
        combined += _make_gripper_mesh_visual(
            gripper_mesh_template,
            g.pose_4x4,
            _confidence_color(g.confidence),
        )

    combined.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_path), combined)


def _build_gripper_lines(transform, gripper_name):
    from grasp_gen.robot import load_control_points_for_visualization

    ctrl_pts_list = load_control_points_for_visualization(gripper_name)
    lines = []
    for ctrl_pts in ctrl_pts_list:
        pts = np.array(ctrl_pts, dtype=np.float32)
        pts_h = np.hstack([pts, np.ones([len(pts), 1])])
        transformed = (transform[:3, :] @ pts_h.T).T
        lines.append(transformed)
    return lines


def _save_gripper_lines(candidates, gripper_name, out_path: Path, max_grasps: int):
    if not candidates or max_grasps <= 0:
        return

    import open3d as o3d

    points = []
    lines = []
    colors = []

    for g in candidates[:max_grasps]:
        color = _confidence_color(g.confidence)
        for line_pts in _build_gripper_lines(g.pose_4x4, gripper_name):
            start = len(points)
            points.extend(np.asarray(line_pts, dtype=np.float32).tolist())
            lines.extend([[start + i, start + i + 1] for i in range(len(line_pts) - 1)])
            colors.extend([color] * (len(line_pts) - 1))

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(str(out_path), line_set)


def _sample_preview_points(pts, colors, max_points: int):
    if max_points <= 0 or len(pts) <= max_points:
        return pts, colors
    indices = np.linspace(0, len(pts) - 1, max_points, dtype=np.int64)
    return pts[indices], colors[indices]


def _preview_bounds(pts, candidates):
    parts = []
    if len(pts):
        parts.append(np.asarray(pts, dtype=np.float32))
    for g in candidates:
        parts.append(np.asarray(g.pose_4x4[:3, 3], dtype=np.float32).reshape(1, 3))
    if not parts:
        return np.zeros(3, dtype=np.float32), 0.1

    cloud = np.vstack(parts)
    mins = cloud.min(axis=0)
    maxs = cloud.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.6)
    return center, max(radius, 0.05)


def _set_axes_equal(ax, pts, candidates):
    center, radius = _preview_bounds(pts, candidates)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] + radius, center[1] - radius)
    ax.set_aspect("equal", adjustable="box")


def _split_interactive_points(object_pts, object_colors, scene_pts, scene_colors, max_points: int):
    if max_points <= 0:
        return object_pts, object_colors, scene_pts, scene_colors

    object_limit = min(len(object_pts), max_points)
    scene_limit = max(0, max_points - object_limit)
    object_pts, object_colors = _sample_preview_points(object_pts, object_colors, object_limit)
    scene_pts, scene_colors = _sample_preview_points(scene_pts, scene_colors, scene_limit)
    return object_pts, object_colors, scene_pts, scene_colors


def _save_interactive_preview(
    object_pts,
    object_colors,
    scene_pts,
    scene_colors,
    candidates,
    gripper_name: str,
    out_path: Path,
    max_grasps: int,
    max_points: int,
    show_scene: bool,
):
    import plotly.graph_objects as go

    if not show_scene:
        scene_pts = scene_pts[:0]
        scene_colors = scene_colors[:0]

    object_pts, object_colors, scene_pts, scene_colors = _split_interactive_points(
        object_pts,
        object_colors,
        scene_pts,
        scene_colors,
        max_points,
    )

    traces = []
    if len(scene_pts):
        traces.append(
            go.Scatter3d(
                x=scene_pts[:, 0],
                y=scene_pts[:, 1],
                z=scene_pts[:, 2],
                mode="markers",
                name=f"scene ({len(scene_pts)})",
                marker={"size": 1.2, "color": "rgba(160,160,160,0.25)"},
                hoverinfo="skip",
            )
        )
    if len(object_pts):
        traces.append(
            go.Scatter3d(
                x=object_pts[:, 0],
                y=object_pts[:, 1],
                z=object_pts[:, 2],
                mode="markers",
                name=f"object ({len(object_pts)})",
                marker={"size": 2.0, "color": "rgb(0,220,0)"},
                hoverinfo="skip",
            )
        )

    positions = []
    labels = []
    hover = []
    marker_colors = []
    for i, g in enumerate(candidates[:max_grasps]):
        color = _confidence_color(g.confidence)
        plot_color = _rgb_string(color)
        for line_pts in _build_gripper_lines(g.pose_4x4, gripper_name):
            line_pts = np.asarray(line_pts, dtype=np.float32)
            traces.append(
                go.Scatter3d(
                    x=line_pts[:, 0],
                    y=line_pts[:, 1],
                    z=line_pts[:, 2],
                    mode="lines",
                    name=f"gripper #{i}",
                    line={"color": plot_color, "width": 5},
                    hoverinfo="skip",
                    showlegend=i == 0,
                )
            )
        pos = np.asarray(g.pose_4x4[:3, 3], dtype=np.float32)
        positions.append(pos)
        labels.append(f"#{i}")
        hover.append(f"#{i}<br>confidence={g.confidence:.4f}<br>x={pos[0]:.4f}m<br>y={pos[1]:.4f}m<br>z={pos[2]:.4f}m")
        marker_colors.append(plot_color)

    if positions:
        positions = np.vstack(positions)
        traces.append(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers+text",
                name="grasp poses",
                text=labels,
                textposition="top center",
                hovertext=hover,
                hoverinfo="text",
                marker={"size": 5, "color": marker_colors},
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Interactive Grasp Preview",
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m, camera-down)",
            "zaxis_title": "Z (m)",
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 0.0, "y": -0.55, "z": -1.8},
                "up": {"x": 0.0, "y": -1.0, "z": 0.0},
            },
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        showlegend=True,
    )
    fig.write_html(
        str(out_path),
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False},
    )


def _save_labeled_preview(
    pts,
    colors,
    candidates,
    gripper_name: str,
    out_path: Path,
    max_grasps: int,
    width: int,
    height: int,
    render_labels: bool,
):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)

    if len(pts):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=colors.astype(np.float64) / 255.0,
            s=0.45,
            alpha=0.70,
        )

    for i, g in enumerate(candidates[:max_grasps]):
        color = _confidence_color(g.confidence)
        for line_pts in _build_gripper_lines(g.pose_4x4, gripper_name):
            line_pts = np.asarray(line_pts, dtype=np.float32)
            ax.plot(
                line_pts[:, 0],
                line_pts[:, 1],
                color=color,
                linewidth=1.4,
            )
        pos = g.pose_4x4[:3, 3]
        if render_labels:
            ax.annotate(
                f"#{i} conf={g.confidence:.3f}\nz={pos[2]:.3f}m",
                xy=(float(pos[0]), float(pos[1])),
                xytext=(4, 4),
                textcoords="offset points",
                color=color,
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.75},
            )
        ax.scatter(
            float(pos[0]),
            float(pos[1]),
            c=[color],
            s=16,
        )

    _set_axes_equal(ax, pts, candidates[:max_grasps])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m, camera-down)")
    title = "Grasp preview projection: color = confidence"
    if render_labels:
        title += ", label = index/confidence/depth"
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _render_debug_previews(
    *,
    pts,
    colors,
    object_pts,
    object_colors,
    scene_pts,
    scene_colors,
    candidates,
    gripper_name: str,
    out_dir: Path,
    max_grasps: int,
    width: int,
    height: int,
    show_scene: bool,
    render_labels: bool,
    render_interactive: bool,
    interactive_max_points: int,
    interactive_show_scene: bool,
    logger,
    file_prefix: str = "grasp",
):
    labeled_preview_name = f"{file_prefix}_preview_labeled.png"
    interactive_name = f"{file_prefix}_preview.html"
    metadata_name = f"{file_prefix}_preview_meta.json"
    metadata = {
        "status": "ok",
        "file_prefix": file_prefix,
        "show_scene": show_scene,
        "render_labels": render_labels,
        "render_interactive": render_interactive,
        "interactive_show_scene": interactive_show_scene,
        "interactive_max_points": int(interactive_max_points),
        "point_count": int(len(pts)),
        "object_points": int(len(object_pts)),
        "scene_points": int(len(scene_pts)),
        "max_grasps": int(max_grasps),
        "images": [],
        "interactive": [],
        "errors": [],
    }

    if render_labels:
        try:
            _save_labeled_preview(
                pts,
                colors,
                candidates,
                gripper_name,
                out_dir / labeled_preview_name,
                max_grasps,
                width,
                height,
                True,
            )
            metadata["images"].append(labeled_preview_name)
        except Exception as exc:
            metadata["errors"].append(f"labeled_preview: {exc}")

    if render_interactive:
        try:
            _save_interactive_preview(
                object_pts,
                object_colors,
                scene_pts,
                scene_colors,
                candidates,
                gripper_name,
                out_dir / interactive_name,
                max_grasps,
                interactive_max_points,
                interactive_show_scene,
            )
            metadata["interactive"].append(interactive_name)
        except Exception as exc:
            metadata["errors"].append(f"interactive_preview: {exc}")

    if metadata["errors"]:
        metadata["status"] = "partial" if metadata["images"] else "failed"

    (out_dir / metadata_name).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if metadata["status"] == "failed":
        logger.warning(f"Failed to render grasp debug previews: {metadata['errors']}")
    else:
        logger.info(f"Rendered grasp debug previews in {out_dir}")


def _write_full_debug_artifacts(
    *,
    object_pts,
    object_colors,
    object_raw_pts,
    object_raw_colors,
    object_graspgen_input_pts,
    object_graspgen_input_colors,
    scene_pts,
    scene_colors,
    candidates,
    gripper_name: str,
    out_dir: Path,
    max_grasps: int,
    save_completion_clouds: bool,
    render_enabled: bool,
    render_max_points: int,
    width: int,
    height: int,
    show_scene: bool,
    render_labels: bool,
    render_interactive: bool,
    interactive_max_points: int,
    interactive_show_scene: bool,
    logger,
):
    try:
        all_pts = np.vstack([object_pts, scene_pts]).astype(np.float32)
        all_colors = np.vstack([object_colors, scene_colors]).astype(np.uint8)
        _save_ply(object_pts, object_colors, out_dir / "object_cloud.ply")
        if save_completion_clouds:
            _save_ply(object_raw_pts, object_raw_colors, out_dir / "object_cloud_raw.ply")
            _save_ply(object_pts, object_colors, out_dir / "object_cloud_completed.ply")
            _save_ply(
                object_graspgen_input_pts,
                object_graspgen_input_colors,
                out_dir / "object_cloud_graspgen_input.ply",
            )
        _save_ply(scene_pts, scene_colors, out_dir / "scene_cloud.ply")
        _save_ply(all_pts, all_colors, out_dir / "grasp_cloud.ply")
        _save_gripper_meshes(candidates, gripper_name, out_dir / "grasp_grippers.ply", max_grasps)
        _save_gripper_lines(candidates, gripper_name, out_dir / "grasp_lines.ply", max_grasps)
    except Exception as exc:
        logger.warning(f"Failed to write full grasp debug artifacts in {out_dir}: {exc}")
        return

    if render_enabled:
        render_pts = all_pts if show_scene else object_pts
        render_colors = all_colors if show_scene else object_colors
        render_pts, render_colors = _sample_preview_points(render_pts, render_colors, render_max_points)
        _render_debug_previews(
            pts=render_pts,
            colors=render_colors,
            object_pts=object_pts,
            object_colors=object_colors,
            scene_pts=scene_pts,
            scene_colors=scene_colors,
            candidates=candidates,
            gripper_name=gripper_name,
            out_dir=out_dir,
            max_grasps=max_grasps,
            width=width,
            height=height,
            show_scene=show_scene,
            render_labels=render_labels,
            render_interactive=render_interactive,
            interactive_max_points=interactive_max_points,
            interactive_show_scene=interactive_show_scene,
            logger=logger,
        )
    logger.info(f"Finished full grasp debug outputs in {out_dir}")


class GraspPlannerNode(Node):
    def __init__(self):
        super().__init__("grasp_planner")

        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self.declare_parameter("device", "cuda")
        self.declare_parameter("gripper_config", "graspgen_robotiq_2f_140.yml")
        self.declare_parameter("model_dir", "")
        self.declare_parameter("depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/wrist/aligned_depth_to_color/camera_info")
        self.declare_parameter("detect_service", "/grounded_sam2/detect_and_segment")
        self.declare_parameter("grasp_threshold", 0.5)
        self.declare_parameter("num_grasps", 2000)
        self.declare_parameter("topk_num_grasps", 300)
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("enable_collision_filter", True)
        self.declare_parameter("collision_threshold", 0.005)
        self.declare_parameter("collision_gripper", "")
        self.declare_parameter("enable_tabletop_filter", True)
        self.declare_parameter(
            "enable_source_gripper_tabletop_sweep",
            DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP,
        )
        self.declare_parameter("require_tabletop_filter", True)
        self.declare_parameter("tabletop_clearance", 0.003)
        self.declare_parameter("tabletop_pregrasp_distance", 0.08)
        self.declare_parameter("tabletop_pregrasp_steps", 5)
        self.declare_parameter("tabletop_ransac_threshold", 0.006)
        self.declare_parameter("tabletop_min_inlier_ratio", 0.15)
        self.declare_parameter("tabletop_filter_mode", "strict")
        self.declare_parameter("adaptive_tabletop_low_profile_height", 0.035)
        self.declare_parameter("adaptive_tabletop_clearance_min", 0.001)
        self.declare_parameter("adaptive_tabletop_clearance_max", 0.002)
        self.declare_parameter("adaptive_tabletop_clearance_height_ratio", 0.12)
        self.declare_parameter("adaptive_tabletop_pregrasp_min", 0.02)
        self.declare_parameter("adaptive_tabletop_pregrasp_height_ratio", 2.0)
        self.declare_parameter("adaptive_tabletop_hard_floor", -0.002)
        self.declare_parameter("adaptive_tabletop_auto_tune", True)
        self.declare_parameter("adaptive_tabletop_retry_clearances", "0.002,0.001")
        self.declare_parameter("adaptive_tabletop_retry_pregrasp_height_ratio", 1.5)
        self.declare_parameter("adaptive_tabletop_retry_hard_floor", -0.003)
        self.declare_parameter("target_width_axis_local", "1.0,0.0,0.0")
        self.declare_parameter("target_width_percentile_low", 5.0)
        self.declare_parameter("target_width_percentile_high", 95.0)
        self.declare_parameter("target_width_min_m", 0.005)
        self.declare_parameter("target_width_max_m", 0.14)
        self.declare_parameter("enable_object_cloud_completion", True)
        self.declare_parameter("object_cloud_completion_mode", "mask_depth_inpaint")
        self.declare_parameter("object_cloud_completion_max_points", 5000)
        self.declare_parameter("object_cloud_completion_kernel_size", 5)
        self.declare_parameter("object_cloud_completion_min_neighbors", 6)
        self.declare_parameter("enable_object_cloud_prismatic_extrude", True)
        self.declare_parameter("object_cloud_prismatic_extrude_max_points", 8000)
        self.declare_parameter("object_cloud_prismatic_extrude_layers", 8)
        self.declare_parameter("enable_scene_cloud_table_holes", True)
        self.declare_parameter("scene_cloud_table_holes_max_points", 8000)
        self.declare_parameter("input_buffer_size", 30)
        self.declare_parameter("sync_max_age_sec", 0.20)
        self.declare_parameter("save_debug_outputs", False)
        self.declare_parameter("debug_output_dir", "outputs/grasp_pipeline")
        self.declare_parameter("debug_max_save_grasps", 10)
        self.declare_parameter("debug_render_preview", True)
        self.declare_parameter("debug_render_show_scene", False)
        self.declare_parameter("debug_render_labels", True)
        self.declare_parameter("debug_render_max_points", 60000)
        self.declare_parameter("debug_render_width", 1280)
        self.declare_parameter("debug_render_height", 720)
        self.declare_parameter("debug_render_interactive", True)
        self.declare_parameter("debug_interactive_show_scene", True)
        self.declare_parameter("debug_interactive_max_points", 120000)

        buffer_size = int(self.get_parameter("input_buffer_size").get_parameter_value().integer_value)
        self._depth_buffer: deque[DepthFrame] = deque(maxlen=max(1, buffer_size))
        self._info_buffer: deque[CameraInfoFrame] = deque(maxlen=max(1, buffer_size))

        device = self.get_parameter("device").get_parameter_value().string_value
        gripper_config = self.get_parameter("gripper_config").get_parameter_value().string_value
        model_dir = self.get_parameter("model_dir").get_parameter_value().string_value or None
        collision_gripper = self.get_parameter("collision_gripper").get_parameter_value().string_value or None

        self.get_logger().info(f"Loading GraspGen models (gripper={gripper_config}) ...")
        self._wrapper = GraspGenWrapper(
            gripper_config=gripper_config,
            model_dir=model_dir,
            device=device,
            collision_gripper=collision_gripper,
        )
        self.get_logger().info(f"GraspGen models loaded (collision_gripper={self._wrapper.collision_gripper_name}).")

        cb_group = ReentrantCallbackGroup()
        detect_cb_group = ReentrantCallbackGroup()
        srv_cb_group = MutuallyExclusiveCallbackGroup()

        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        detect_service = self.get_parameter("detect_service").get_parameter_value().string_value

        self._depth_sub = self.create_subscription(
            Image, depth_topic, self._depth_cb, qos_profile_sensor_data, callback_group=cb_group
        )
        self._info_sub = self.create_subscription(
            CameraInfo, info_topic, self._info_cb, qos_profile_sensor_data, callback_group=cb_group
        )

        self._detect_client = self.create_client(DetectSegment, detect_service, callback_group=detect_cb_group)

        self._srv = self.create_service(PlanGrasp, "~/plan_grasp", self._srv_cb, callback_group=srv_cb_group)

        self._grasp_pub = self.create_publisher(GraspCandidateArray, "~/grasps", 10)

        self.get_logger().info(
            f"GraspPlannerNode ready — depth: {depth_topic}, info: {info_topic}, detect: {detect_service}"
        )

    def _depth_cb(self, msg: Image):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            self.get_logger().warn("Failed to convert depth image")
            return
        fallback_scale = self.get_parameter("depth_scale").get_parameter_value().double_value
        frame = DepthFrame(
            stamp_ns=_stamp_to_ns(msg.header.stamp),
            frame_id=msg.header.frame_id,
            data=depth,
            depth_scale=_depth_scale_for_msg(msg, fallback_scale),
        )
        with self._lock:
            self._depth_buffer.append(frame)

    def _info_cb(self, msg: CameraInfo):
        frame = CameraInfoFrame(stamp_ns=_stamp_to_ns(msg.header.stamp), msg=msg)
        with self._lock:
            self._info_buffer.append(frame)

    @staticmethod
    def _nearest_by_stamp(frames, target_stamp_ns: int):
        if not frames:
            return None, None
        if target_stamp_ns == 0 or all(frame.stamp_ns == 0 for frame in frames):
            return frames[-1], 0
        frame = min(frames, key=lambda f: abs(f.stamp_ns - target_stamp_ns))
        return frame, abs(frame.stamp_ns - target_stamp_ns)

    def _get_synchronized_inputs(self, target_stamp_ns: int):
        max_age_ns = int(self.get_parameter("sync_max_age_sec").get_parameter_value().double_value * 1_000_000_000)
        with self._lock:
            depth_frames = list(self._depth_buffer)
            info_frames = list(self._info_buffer)

        depth_frame, depth_age = self._nearest_by_stamp(depth_frames, target_stamp_ns)
        info_frame, info_age = self._nearest_by_stamp(info_frames, target_stamp_ns)
        if depth_frame is None or info_frame is None:
            return None
        if depth_age is not None and depth_age > max_age_ns:
            self.get_logger().warn(f"No synchronized depth frame: age {depth_age / 1e9:.3f}s > {max_age_ns / 1e9:.3f}s")
            return None
        if info_age is not None and info_age > max_age_ns:
            self.get_logger().warn(f"No synchronized CameraInfo: age {info_age / 1e9:.3f}s > {max_age_ns / 1e9:.3f}s")
            return None
        return depth_frame, info_frame.msg

    def _get_segmentation_mask(self, text_prompt: str, confidence_threshold: float):
        if not self._detect_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("DetectSegment service not available")
            return None, "detect_service_unavailable"

        request = DetectSegment.Request()
        request.text_prompt = text_prompt
        request.confidence_threshold = confidence_threshold

        future = self._detect_client.call_async(request)
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        if not done_event.wait(timeout=10.0):
            self.get_logger().error("Timed out waiting for DetectSegment response")
            return None, "detect_service_timeout"

        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"DetectSegment service call failed: {exc}")
            return None, f"detect_service_error: {exc}"
        if result is None:
            return None, "detect_service_no_result"
        if not result.success:
            return None, f"detect_service_failed: {result.message}"
        if len(result.detections.detections) == 0:
            return (
                None,
                f"no_detections: target '{text_prompt}' not found at confidence_threshold={confidence_threshold:.2f}",
            )
        best_det = max(result.detections.detections, key=lambda d: d.confidence)
        if best_det.confidence < confidence_threshold:
            return (
                None,
                f"low_confidence: best detection confidence={best_det.confidence:.4f} < threshold={confidence_threshold:.2f}",
            )
        mask = self._bridge.imgmsg_to_cv2(best_det.mask, desired_encoding="mono8")
        return (mask, best_det.mask.header, best_det.confidence, best_det), None

    def _resolve_debug_output_mode(self, request_mode: str) -> str:
        mode = _normalize_debug_output_mode(request_mode)
        if mode != _DEBUG_OUTPUT_DEFAULT:
            return mode
        if self.get_parameter("save_debug_outputs").get_parameter_value().bool_value:
            return _DEBUG_OUTPUT_FULL
        return _DEBUG_OUTPUT_NONE

    def _new_debug_output_dir(self, text_prompt: str) -> Path:
        output_root = Path(self.get_parameter("debug_output_dir").get_parameter_value().string_value)
        if not output_root.is_absolute():
            output_root = Path.cwd() / output_root

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = output_root / f"{stamp}_{_safe_name(text_prompt)}_{time.time_ns()}"
        out_dir.mkdir(parents=True, exist_ok=False)
        return out_dir

    def _save_failure_diagnostic_json(
        self,
        *,
        text_prompt: str,
        confidence_threshold: float,
        grasp_threshold: float,
        output_mode: str,
        message: str,
        diagnostic_details: list[str],
        elapsed_ms: float = 0.0,
    ) -> Path:
        out_dir = self._new_debug_output_dir(text_prompt)
        result = {
            "prompt": text_prompt,
            "confidence_threshold": confidence_threshold,
            "grasp_threshold": grasp_threshold,
            "inference_time_ms": round(float(elapsed_ms), 3),
            "num_grasps": 0,
            "success": False,
            "message": message,
            "debug_output_mode": output_mode,
            "diagnostic_details": diagnostic_details,
            "pointcloud": None,
            "render": {"scheduled": False},
            "grasps": [],
        }
        (out_dir / "grasp_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out_dir

    def _populate_failure_response(
        self,
        response: PlanGrasp.Response,
        *,
        text_prompt: str,
        confidence_threshold: float,
        grasp_threshold: float,
        output_mode: str,
        message: str,
        diagnostic_details: list[str],
        elapsed_ms: float = 0.0,
    ):
        response.success = False
        response.message = message
        response.inference_time_ms = elapsed_ms
        response.diagnostic_details = [*diagnostic_details, f"debug_output_mode: {output_mode}"]
        if _debug_output_enabled(output_mode):
            try:
                out_dir = self._save_failure_diagnostic_json(
                    text_prompt=text_prompt,
                    confidence_threshold=confidence_threshold,
                    grasp_threshold=grasp_threshold,
                    output_mode=output_mode,
                    message=message,
                    diagnostic_details=response.diagnostic_details,
                    elapsed_ms=elapsed_ms,
                )
                response.debug_output_dir = str(out_dir)
                response.diagnostic_details.append(f"debug_output_dir: {out_dir}")
            except Exception as exc:
                self.get_logger().warn(f"Failed to save failure diagnostic JSON: {exc}")
                response.diagnostic_details.append(f"debug_output_error: {exc}")
        return response

    def _save_debug_outputs(
        self,
        *,
        text_prompt: str,
        confidence_threshold: float,
        grasp_threshold: float,
        depth_frame: DepthFrame,
        camera_info: CameraInfo,
        binary_mask: np.ndarray,
        camera_intrinsics: np.ndarray,
        candidates,
        diagnostic: GraspDiagnostic,
        detection_confidence: float,
        detection_stamp_ns: int,
        elapsed_ms: float,
        output_mode: str,
    ):
        out_dir = self._new_debug_output_dir(text_prompt)

        result = {
            "prompt": text_prompt,
            "confidence_threshold": confidence_threshold,
            "grasp_threshold": grasp_threshold,
            "inference_time_ms": round(float(elapsed_ms), 3),
            "num_grasps": len(candidates),
            "debug_output_mode": output_mode,
            "diagnostic": _diagnostic_to_dict(diagnostic, detection_confidence=detection_confidence),
            "depth_frame_id": depth_frame.frame_id,
            "camera_frame_id": camera_info.header.frame_id,
            "detection_stamp_ns": int(detection_stamp_ns),
            "depth_stamp_ns": int(depth_frame.stamp_ns),
            "camera_info_stamp_ns": _stamp_to_ns(camera_info.header.stamp),
            "geometry_stamp_ns": int(depth_frame.stamp_ns or detection_stamp_ns),
            "depth_detection_delta_ms": round(abs(depth_frame.stamp_ns - detection_stamp_ns) / 1_000_000.0, 3),
            "depth_scale": depth_frame.depth_scale,
            "camera_info": {
                "k": list(camera_info.k),
                "width": camera_info.width,
                "height": camera_info.height,
            },
            "gripper": self._wrapper.gripper_name,
            "visualization_gripper": getattr(self._wrapper, "collision_gripper_name", self._wrapper.gripper_name),
            "pointcloud": None,
            "render": {"scheduled": False},
            "grasps": [],
        }
        for i, g in enumerate(candidates):
            result["grasps"].append(_candidate_debug_record(i, g))

        if output_mode == _DEBUG_OUTPUT_DIAGNOSTIC:
            (out_dir / "grasp_result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return out_dir

        object_raw_pts = _debug_points_or_fallback(diagnostic.object_pc_raw, np.zeros((0, 3), dtype=np.float32))
        object_raw_colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (len(object_raw_pts), 1))
        object_pts = _debug_points_or_fallback(diagnostic.object_pc_after_completion, object_raw_pts)
        object_colors = _object_stage_colors(
            len(object_pts),
            diagnostic.object_point_count_raw,
            diagnostic.object_completion_added_count,
        )
        object_graspgen_input_pts = _debug_points_or_fallback(diagnostic.object_pc_inference_input, object_pts)
        object_graspgen_input_colors = np.tile(
            np.array([[0, 200, 255]], dtype=np.uint8),
            (len(object_graspgen_input_pts), 1),
        )
        scene_pts = _debug_points_or_fallback(diagnostic.scene_pc_after_completion, np.zeros((0, 3), dtype=np.float32))
        scene_pts_raw_count = int(diagnostic.scene_point_count_raw) or len(scene_pts)
        scene_colors = _scene_stage_colors(len(scene_pts), scene_pts_raw_count)

        max_grasps = int(self.get_parameter("debug_max_save_grasps").get_parameter_value().integer_value)
        vis_gripper = getattr(self._wrapper, "collision_gripper_name", self._wrapper.gripper_name)

        render_enabled = self.get_parameter("debug_render_preview").get_parameter_value().bool_value
        result["pointcloud"] = {
            "object_cloud": "object_cloud.ply",
            "object_cloud_raw": "object_cloud_raw.ply"
            if diagnostic.object_cloud_completion_enabled or diagnostic.object_prismatic_extrude_enabled
            else None,
            "object_cloud_completed": "object_cloud_completed.ply"
            if diagnostic.object_cloud_completion_enabled or diagnostic.object_prismatic_extrude_enabled
            else None,
            "object_cloud_graspgen_input": "object_cloud_graspgen_input.ply"
            if diagnostic.object_cloud_completion_enabled or diagnostic.object_prismatic_extrude_enabled
            else None,
            "scene_cloud": "scene_cloud.ply",
            "grasp_cloud": "grasp_cloud.ply",
            "object_points": len(object_pts),
            "object_points_raw": len(object_raw_pts),
            "object_points_graspgen_input": len(object_graspgen_input_pts),
            "scene_points": len(scene_pts),
        }
        result["render"] = {
            "scheduled": bool(render_enabled),
            "labeled_preview": "grasp_preview_labeled.png",
            "interactive_preview": "grasp_preview.html",
            "metadata": "grasp_preview_meta.json",
        }
        (out_dir / "grasp_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        show_scene = self.get_parameter("debug_render_show_scene").get_parameter_value().bool_value
        render_labels = self.get_parameter("debug_render_labels").get_parameter_value().bool_value
        render_interactive = self.get_parameter("debug_render_interactive").get_parameter_value().bool_value
        interactive_show_scene = self.get_parameter("debug_interactive_show_scene").get_parameter_value().bool_value
        width = int(self.get_parameter("debug_render_width").get_parameter_value().integer_value)
        height = int(self.get_parameter("debug_render_height").get_parameter_value().integer_value)
        interactive_max_points = int(
            self.get_parameter("debug_interactive_max_points").get_parameter_value().integer_value
        )
        render_max_points = int(self.get_parameter("debug_render_max_points").get_parameter_value().integer_value)
        debug_candidates = [
            DebugPreviewCandidate(
                pose_4x4=np.array(g.pose_4x4, dtype=np.float32, copy=True),
                confidence=float(g.confidence),
                collision_free=bool(g.collision_free),
            )
            for g in candidates[:max_grasps]
        ]
        thread = threading.Thread(
            target=_write_full_debug_artifacts,
            kwargs={
                "object_pts": np.array(object_pts, dtype=np.float32, copy=True),
                "object_colors": np.array(object_colors, dtype=np.uint8, copy=True),
                "object_raw_pts": np.array(object_raw_pts, dtype=np.float32, copy=True),
                "object_raw_colors": np.array(object_raw_colors, dtype=np.uint8, copy=True),
                "object_graspgen_input_pts": np.array(object_graspgen_input_pts, dtype=np.float32, copy=True),
                "object_graspgen_input_colors": np.array(object_graspgen_input_colors, dtype=np.uint8, copy=True),
                "scene_pts": np.array(scene_pts, dtype=np.float32, copy=True),
                "scene_colors": np.array(scene_colors, dtype=np.uint8, copy=True),
                "candidates": debug_candidates,
                "gripper_name": vis_gripper,
                "out_dir": out_dir,
                "max_grasps": max_grasps,
                "save_completion_clouds": bool(
                    diagnostic.object_cloud_completion_enabled or diagnostic.object_prismatic_extrude_enabled
                ),
                "render_enabled": render_enabled,
                "render_max_points": render_max_points,
                "width": max(320, width),
                "height": max(240, height),
                "show_scene": show_scene,
                "render_labels": render_labels,
                "render_interactive": render_interactive,
                "interactive_max_points": interactive_max_points,
                "interactive_show_scene": interactive_show_scene,
                "logger": self.get_logger(),
            },
            name="grasp-debug-output",
            daemon=True,
        )
        thread.start()

        return out_dir

    def _srv_cb(self, request: PlanGrasp.Request, response: PlanGrasp.Response):
        try:
            debug_output_mode = self._resolve_debug_output_mode(request.debug_output_mode)
        except ValueError as exc:
            response.success = False
            response.message = str(exc)
            response.diagnostic_details = ["failure_stage: input", f"failure_reason: {exc}"]
            response.inference_time_ms = 0.0
            return response

        confidence_threshold = request.confidence_threshold if request.confidence_threshold > 0 else 0.25
        grasp_threshold = (
            request.grasp_threshold
            if request.grasp_threshold > 0
            else self.get_parameter("grasp_threshold").get_parameter_value().double_value
        )

        if not request.text_prompt:
            return self._populate_failure_response(
                response,
                text_prompt=request.text_prompt,
                confidence_threshold=confidence_threshold,
                grasp_threshold=grasp_threshold,
                output_mode=debug_output_mode,
                message="Empty text_prompt",
                diagnostic_details=["failure_stage: input", "failure_reason: text_prompt is empty"],
            )

        self.get_logger().info(f"Getting segmentation for '{request.text_prompt}' ...")
        segmentation_result, detect_failure = self._get_segmentation_mask(request.text_prompt, confidence_threshold)
        if detect_failure is not None or segmentation_result is None:
            reason = detect_failure or "detect_service_no_result"
            return self._populate_failure_response(
                response,
                text_prompt=request.text_prompt,
                confidence_threshold=confidence_threshold,
                grasp_threshold=grasp_threshold,
                output_mode=debug_output_mode,
                message=f"Detection failed: {reason}",
                diagnostic_details=[
                    "failure_stage: detection",
                    f"failure_reason: {reason}",
                    f"text_prompt: {request.text_prompt}",
                    f"confidence_threshold: {confidence_threshold:.2f}",
                ],
            )
        mask, mask_header, best_confidence, best_detection = segmentation_result

        detection_stamp_ns = _stamp_to_ns(mask_header.stamp)
        synced_inputs = self._get_synchronized_inputs(detection_stamp_ns)
        if synced_inputs is None:
            return self._populate_failure_response(
                response,
                text_prompt=request.text_prompt,
                confidence_threshold=confidence_threshold,
                grasp_threshold=grasp_threshold,
                output_mode=debug_output_mode,
                message="No synchronized depth/CameraInfo for detection frame",
                diagnostic_details=[
                    "failure_stage: sync",
                    "failure_reason: depth or CameraInfo not available within sync tolerance",
                ],
            )
        depth_frame, info = synced_inputs

        binary_mask = (mask > 0).astype(np.int32)

        K = np.array(info.k).reshape(3, 3)

        t0 = time.time()
        try:
            candidates, diag = self._wrapper.plan_grasps(
                depth_image=depth_frame.data,
                segmentation_mask=binary_mask,
                camera_intrinsics=K,
                depth_scale=depth_frame.depth_scale,
                grasp_threshold=grasp_threshold,
                num_grasps=int(self.get_parameter("num_grasps").get_parameter_value().integer_value),
                topk_num_grasps=int(self.get_parameter("topk_num_grasps").get_parameter_value().integer_value),
                enable_collision_filter=self.get_parameter("enable_collision_filter").get_parameter_value().bool_value,
                collision_threshold=self.get_parameter("collision_threshold").get_parameter_value().double_value,
                enable_tabletop_filter=self.get_parameter("enable_tabletop_filter").get_parameter_value().bool_value,
                enable_source_gripper_tabletop_sweep=self.get_parameter("enable_source_gripper_tabletop_sweep")
                .get_parameter_value()
                .bool_value,
                require_tabletop_filter=self.get_parameter("require_tabletop_filter").get_parameter_value().bool_value,
                tabletop_clearance=self.get_parameter("tabletop_clearance").get_parameter_value().double_value,
                tabletop_pregrasp_distance=self.get_parameter("tabletop_pregrasp_distance")
                .get_parameter_value()
                .double_value,
                tabletop_pregrasp_steps=int(
                    self.get_parameter("tabletop_pregrasp_steps").get_parameter_value().integer_value
                ),
                tabletop_ransac_threshold=self.get_parameter("tabletop_ransac_threshold")
                .get_parameter_value()
                .double_value,
                tabletop_min_inlier_ratio=self.get_parameter("tabletop_min_inlier_ratio")
                .get_parameter_value()
                .double_value,
                tabletop_filter_mode=self.get_parameter("tabletop_filter_mode").get_parameter_value().string_value,
                adaptive_tabletop_low_profile_height=self.get_parameter("adaptive_tabletop_low_profile_height")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_clearance_min=self.get_parameter("adaptive_tabletop_clearance_min")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_clearance_max=self.get_parameter("adaptive_tabletop_clearance_max")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_clearance_height_ratio=self.get_parameter("adaptive_tabletop_clearance_height_ratio")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_pregrasp_min=self.get_parameter("adaptive_tabletop_pregrasp_min")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_pregrasp_height_ratio=self.get_parameter("adaptive_tabletop_pregrasp_height_ratio")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_hard_floor=self.get_parameter("adaptive_tabletop_hard_floor")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_auto_tune=self.get_parameter("adaptive_tabletop_auto_tune")
                .get_parameter_value()
                .bool_value,
                adaptive_tabletop_retry_clearances=_parse_float_csv(
                    self.get_parameter("adaptive_tabletop_retry_clearances").get_parameter_value().string_value
                ),
                adaptive_tabletop_retry_pregrasp_height_ratio=self.get_parameter(
                    "adaptive_tabletop_retry_pregrasp_height_ratio"
                )
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_retry_hard_floor=self.get_parameter("adaptive_tabletop_retry_hard_floor")
                .get_parameter_value()
                .double_value,
                target_width_axis_local=_parse_vector3_parameter(
                    self.get_parameter("target_width_axis_local").get_parameter_value().string_value,
                    "target_width_axis_local",
                ),
                target_width_percentile_low=self.get_parameter("target_width_percentile_low")
                .get_parameter_value()
                .double_value,
                target_width_percentile_high=self.get_parameter("target_width_percentile_high")
                .get_parameter_value()
                .double_value,
                target_width_min_m=self.get_parameter("target_width_min_m").get_parameter_value().double_value,
                target_width_max_m=self.get_parameter("target_width_max_m").get_parameter_value().double_value,
                enable_object_cloud_completion=self.get_parameter("enable_object_cloud_completion")
                .get_parameter_value()
                .bool_value,
                object_cloud_completion_mode=self.get_parameter("object_cloud_completion_mode")
                .get_parameter_value()
                .string_value,
                object_cloud_completion_max_points=int(
                    self.get_parameter("object_cloud_completion_max_points").get_parameter_value().integer_value
                ),
                object_cloud_completion_kernel_size=int(
                    self.get_parameter("object_cloud_completion_kernel_size").get_parameter_value().integer_value
                ),
                object_cloud_completion_min_neighbors=int(
                    self.get_parameter("object_cloud_completion_min_neighbors").get_parameter_value().integer_value
                ),
                enable_object_cloud_prismatic_extrude=self.get_parameter("enable_object_cloud_prismatic_extrude")
                .get_parameter_value()
                .bool_value,
                object_cloud_prismatic_extrude_max_points=int(
                    self.get_parameter("object_cloud_prismatic_extrude_max_points").get_parameter_value().integer_value
                ),
                object_cloud_prismatic_extrude_layers=int(
                    self.get_parameter("object_cloud_prismatic_extrude_layers").get_parameter_value().integer_value
                ),
                enable_scene_cloud_table_holes=self.get_parameter("enable_scene_cloud_table_holes")
                .get_parameter_value()
                .bool_value,
                scene_cloud_table_holes_max_points=int(
                    self.get_parameter("scene_cloud_table_holes_max_points").get_parameter_value().integer_value
                ),
            )
        except Exception as exc:
            self.get_logger().error(f"GraspGen error: {exc}")
            return self._populate_failure_response(
                response,
                text_prompt=request.text_prompt,
                confidence_threshold=confidence_threshold,
                grasp_threshold=grasp_threshold,
                output_mode=debug_output_mode,
                message=f"GraspGen inference failed: {exc}",
                diagnostic_details=[
                    "failure_stage: graspgen_inference",
                    f"failure_reason: {exc}",
                ],
            )

        execution_table_parallel = getattr(diag, "_execution_table_fit_thread", None) is not None
        execution_table_started = time.perf_counter()
        execution_table_fit = _fit_execution_table_plane(diag)
        execution_table_wait_s = time.perf_counter() - execution_table_started
        execution_table_holder = getattr(diag, "_execution_table_fit_holder", None)
        execution_table_duration_s = execution_table_wait_s
        if isinstance(execution_table_holder, dict):
            execution_table_duration_s = float(execution_table_holder.get("duration_s", execution_table_duration_s))
        elapsed_ms = (time.time() - t0) * 1000.0

        diagnostic_lines = _diagnostic_to_lines(diag, detection_confidence=best_confidence)
        diagnostic_lines.append(f"final_grasp_count: {len(candidates)}")
        diagnostic_lines.append(f"debug_output_mode: {debug_output_mode}")
        diagnostic_lines.append(f"detection_stamp_ns: {detection_stamp_ns}")
        diagnostic_lines.append(f"depth_stamp_ns: {depth_frame.stamp_ns}")
        diagnostic_lines.append(
            f"depth_detection_delta_ms: {abs(depth_frame.stamp_ns - detection_stamp_ns) / 1_000_000.0:.3f}"
        )
        diagnostic_lines.append(f"execution_table_plane_found: {execution_table_fit.plane is not None}")
        diagnostic_lines.append(f"execution_table_plane_best_inlier_ratio: {execution_table_fit.best_inlier_ratio:.3f}")
        diagnostic_lines.append(f"execution_table_plane_fit_duration_s: {execution_table_duration_s:.6f}")
        diagnostic_lines.append(f"execution_table_plane_wait_duration_s: {execution_table_wait_s:.6f}")
        diagnostic_lines.append(f"execution_table_plane_parallel: {execution_table_parallel}")
        if execution_table_fit.failure_reason:
            diagnostic_lines.append(f"execution_table_plane_failure_reason: {execution_table_fit.failure_reason}")
        if (
            execution_table_fit.plane is not None
            and diag.tabletop_plane_found
            and diag.tabletop_plane_normal is not None
        ):
            normal_dot = float(np.clip(diag.tabletop_plane_normal @ execution_table_fit.plane.normal, -1.0, 1.0))
            plane_angle_deg = float(np.degrees(np.arccos(normal_dot)))
            plane_offset_delta_m = abs(float(diag.tabletop_plane_d) - float(execution_table_fit.plane.d))
            diagnostic_lines.append(f"execution_table_plane_shadow_angle_deg: {plane_angle_deg:.6f}")
            diagnostic_lines.append(f"execution_table_plane_shadow_offset_delta_m: {plane_offset_delta_m:.6f}")
            self.get_logger().info(
                f"Execution table plane fit: compute={execution_table_duration_s * 1000.0:.3f}ms "
                f"wait={execution_table_wait_s * 1000.0:.3f}ms parallel={execution_table_parallel}, "
                f"planning delta angle={plane_angle_deg:.4f}deg offset={plane_offset_delta_m * 1000.0:.3f}mm"
            )
        else:
            self.get_logger().info(
                f"Execution table plane fit: compute={execution_table_duration_s * 1000.0:.3f}ms "
                f"wait={execution_table_wait_s * 1000.0:.3f}ms parallel={execution_table_parallel}"
            )

        if _debug_output_enabled(debug_output_mode):
            try:
                out_dir = self._save_debug_outputs(
                    text_prompt=request.text_prompt,
                    confidence_threshold=confidence_threshold,
                    grasp_threshold=grasp_threshold,
                    depth_frame=depth_frame,
                    camera_info=info,
                    binary_mask=binary_mask,
                    camera_intrinsics=K,
                    candidates=candidates,
                    diagnostic=diag,
                    detection_confidence=best_confidence,
                    detection_stamp_ns=detection_stamp_ns,
                    elapsed_ms=elapsed_ms,
                    output_mode=debug_output_mode,
                )
                response.debug_output_dir = str(out_dir)
                diagnostic_lines.append(f"debug_output_dir: {out_dir}")
                self.get_logger().info(f"Saved grasp debug outputs to {out_dir}")
            except Exception as exc:
                self.get_logger().warn(f"Failed to save grasp debug outputs: {exc}")
                diagnostic_lines.append(f"debug_output_error: {exc}")

        geometry_stamp_ns = depth_frame.stamp_ns or detection_stamp_ns
        header = Header(stamp=_stamp_from_ns(geometry_stamp_ns), frame_id=info.header.frame_id)

        grasp_msgs = []
        for cand in candidates:
            gm = GraspCandidate()
            gm.header = header
            gm.pose_matrix = cand.pose_4x4.flatten().tolist()
            gm.confidence = cand.confidence
            gm.collision_free = cand.collision_free
            gm.target_width_m = float(cand.target_width_m)
            gm.target_width_quality = float(cand.target_width_quality)
            gm.width_axis_camera = [float(v) for v in cand.width_axis_camera]
            gm.target_width_min_offset_m = float(cand.target_width_min_offset_m)
            gm.target_width_max_offset_m = float(cand.target_width_max_offset_m)
            grasp_msgs.append(gm)

        grasp_array = GraspCandidateArray(header=header, grasps=grasp_msgs)
        self._grasp_pub.publish(grasp_array)

        response.grasps = grasp_array
        response.object_centroid_xyz = best_detection.centroid_xyz
        response.object_volume_centroid_xyz = best_detection.volume_centroid_xyz
        response.object_volume_m3 = float(best_detection.volume_m3)
        response.object_point_count = int(best_detection.point_count)
        response.execution_table_plane_found = execution_table_fit.plane is not None
        if execution_table_fit.plane is not None:
            execution_table = execution_table_fit.plane
            response.execution_table_plane_normal.x = float(execution_table.normal[0])
            response.execution_table_plane_normal.y = float(execution_table.normal[1])
            response.execution_table_plane_normal.z = float(execution_table.normal[2])
            response.execution_table_plane_offset = float(execution_table.d)
            response.execution_table_plane_inlier_ratio = float(execution_table.inlier_ratio)
        response.table_plane_found = bool(diag.tabletop_plane_found and diag.tabletop_plane_normal is not None)
        if response.table_plane_found:
            table_normal = diag.tabletop_plane_normal
            assert table_normal is not None
            response.table_plane_normal.x = float(table_normal[0])
            response.table_plane_normal.y = float(table_normal[1])
            response.table_plane_normal.z = float(table_normal[2])
            response.table_plane_offset = float(diag.tabletop_plane_d)
            response.table_plane_inlier_ratio = float(diag.tabletop_inlier_ratio)
        if diag.tabletop_object_top_xyz is not None:
            response.object_top_xyz.x = float(diag.tabletop_object_top_xyz[0])
            response.object_top_xyz.y = float(diag.tabletop_object_top_xyz[1])
            response.object_top_xyz.z = float(diag.tabletop_object_top_xyz[2])
        response.inference_time_ms = elapsed_ms
        response.diagnostic_details = diagnostic_lines

        if len(candidates) == 0:
            response.success = False
            response.message = f"No grasps generated: {diag.failure_reason}"
        else:
            response.success = True
            response.message = f"Generated {len(grasp_msgs)} grasps in {elapsed_ms:.1f}ms"

        self.get_logger().info(
            f"PlanGrasp('{request.text_prompt}'): {len(grasp_msgs)} grasps, {elapsed_ms:.1f}ms, diag={diag.failure_stage or 'ok'}"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspPlannerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
