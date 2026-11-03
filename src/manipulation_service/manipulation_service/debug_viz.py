"""Grasp-debug visualisation: PLY/mesh writers and preview rendering (ROS-free)."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .diagnostics import candidate_debug_record, diagnostic_to_dict
from .grasp_types import GraspDiagnostic

DEBUG_OUTPUT_DEFAULT = "default"
DEBUG_OUTPUT_NONE = "none"
DEBUG_OUTPUT_DIAGNOSTIC = "diagnostic"
DEBUG_OUTPUT_FULL = "full"
VALID_DEBUG_OUTPUT_MODES = {
    "",
    DEBUG_OUTPUT_DEFAULT,
    DEBUG_OUTPUT_NONE,
    DEBUG_OUTPUT_DIAGNOSTIC,
    DEBUG_OUTPUT_FULL,
}


@dataclass
class DebugPreviewCandidate:
    pose_4x4: np.ndarray
    confidence: float
    collision_free: bool


def normalize_debug_output_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in VALID_DEBUG_OUTPUT_MODES:
        valid = ", ".join(sorted(m or "empty" for m in VALID_DEBUG_OUTPUT_MODES))
        raise ValueError(f"invalid debug_output_mode {value!r}; expected one of: {valid}")
    return mode or DEBUG_OUTPUT_DEFAULT


def debug_output_enabled(mode: str) -> bool:
    return mode in {DEBUG_OUTPUT_DIAGNOSTIC, DEBUG_OUTPUT_FULL}


def debug_points_or_fallback(points: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    if points is None:
        return fallback.astype(np.float32, copy=False)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return fallback.astype(np.float32, copy=False)
    return arr


def object_stage_colors(total: int, raw_count: int, inpaint_count: int) -> np.ndarray:
    colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (max(0, int(total)), 1))
    raw_end = min(max(0, int(raw_count)), len(colors))
    inpaint_end = min(raw_end + max(0, int(inpaint_count)), len(colors))
    if inpaint_end > raw_end:
        colors[raw_end:inpaint_end] = np.array([255, 170, 0], dtype=np.uint8)
    if inpaint_end < len(colors):
        colors[inpaint_end:] = np.array([0, 200, 255], dtype=np.uint8)
    return colors


def scene_stage_colors(total: int, raw_count: int) -> np.ndarray:
    colors = np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (max(0, int(total)), 1))
    raw_end = min(max(0, int(raw_count)), len(colors))
    if raw_end < len(colors):
        colors[raw_end:] = np.array([168, 85, 247], dtype=np.uint8)
    return colors


def save_ply(pts, colors, out_path: Path):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(out_path), pcd)


def confidence_color(confidence):
    t = float(np.clip(confidence, 0.0, 1.0))
    return [1.0 - t, t, 0.0]


def rgb_string(color, alpha=None):
    rgb = [int(round(float(c) * 255.0)) for c in color]
    if alpha is None:
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


def make_gripper_mesh_visual(gripper_mesh_template, transform, color):
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


def save_gripper_meshes(candidates, gripper_name, out_path: Path, max_grasps: int, *, skip_empty: bool = True) -> int:
    """Write the top-grasp gripper meshes; return the applied grasp count.

    Single count rule: ``count = min(len(candidates), max(0, int(max_grasps)))``;
    with ``skip_empty`` (default) nothing is written when the count is 0.
    """
    count = min(len(candidates), max(0, int(max_grasps)))
    if skip_empty and count == 0:
        return count

    import open3d as o3d
    from grasp_gen.robot import get_gripper_info

    gripper_info = get_gripper_info(gripper_name)
    gripper_mesh_template = gripper_info.collision_mesh
    combined = o3d.geometry.TriangleMesh()

    for g in candidates[:count]:
        combined += make_gripper_mesh_visual(
            gripper_mesh_template,
            g.pose_4x4,
            confidence_color(g.confidence),
        )

    combined.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_path), combined)
    return count


def build_gripper_lines(transform, gripper_name):
    from grasp_gen.robot import load_control_points_for_visualization

    ctrl_pts_list = load_control_points_for_visualization(gripper_name)
    lines = []
    for ctrl_pts in ctrl_pts_list:
        pts = np.array(ctrl_pts, dtype=np.float32)
        pts_h = np.hstack([pts, np.ones([len(pts), 1])])
        transformed = (transform[:3, :] @ pts_h.T).T
        lines.append(transformed)
    return lines


def save_gripper_lines(candidates, gripper_name, out_path: Path, max_grasps: int, *, skip_empty: bool = True) -> int:
    """Write the top-grasp gripper wireframes; return the applied grasp count.

    Single count rule: ``count = min(len(candidates), max(0, int(max_grasps)))``;
    with ``skip_empty`` (default) nothing is written when the count is 0.
    """
    count = min(len(candidates), max(0, int(max_grasps)))
    if skip_empty and count == 0:
        return count

    import open3d as o3d

    points = []
    lines = []
    colors = []

    for g in candidates[:count]:
        color = confidence_color(g.confidence)
        for line_pts in build_gripper_lines(g.pose_4x4, gripper_name):
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
    return count


def sample_preview_points(pts, colors, max_points: int):
    if max_points <= 0 or len(pts) <= max_points:
        return pts, colors
    indices = np.linspace(0, len(pts) - 1, max_points, dtype=np.int64)
    return pts[indices], colors[indices]


def preview_bounds(pts, candidates):
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


def set_axes_equal(ax, pts, candidates):
    center, radius = preview_bounds(pts, candidates)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] + radius, center[1] - radius)
    ax.set_aspect("equal", adjustable="box")


def split_interactive_points(object_pts, object_colors, scene_pts, scene_colors, max_points: int):
    if max_points <= 0:
        return object_pts, object_colors, scene_pts, scene_colors

    object_limit = min(len(object_pts), max_points)
    scene_limit = max(0, max_points - object_limit)
    object_pts, object_colors = sample_preview_points(object_pts, object_colors, object_limit)
    scene_pts, scene_colors = sample_preview_points(scene_pts, scene_colors, scene_limit)
    return object_pts, object_colors, scene_pts, scene_colors


def save_interactive_preview(
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

    object_pts, object_colors, scene_pts, scene_colors = split_interactive_points(
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
        color = confidence_color(g.confidence)
        plot_color = rgb_string(color)
        for line_pts in build_gripper_lines(g.pose_4x4, gripper_name):
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


def save_labeled_preview(
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
        color = confidence_color(g.confidence)
        for line_pts in build_gripper_lines(g.pose_4x4, gripper_name):
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

    set_axes_equal(ax, pts, candidates[:max_grasps])
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


def render_debug_previews(
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
            save_labeled_preview(
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
            save_interactive_preview(
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


def write_full_debug_artifacts(
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
        save_ply(object_pts, object_colors, out_dir / "object_cloud.ply")
        if save_completion_clouds:
            save_ply(object_raw_pts, object_raw_colors, out_dir / "object_cloud_raw.ply")
            save_ply(object_pts, object_colors, out_dir / "object_cloud_completed.ply")
            save_ply(
                object_graspgen_input_pts,
                object_graspgen_input_colors,
                out_dir / "object_cloud_graspgen_input.ply",
            )
        save_ply(scene_pts, scene_colors, out_dir / "scene_cloud.ply")
        save_ply(all_pts, all_colors, out_dir / "grasp_cloud.ply")
        save_gripper_meshes(candidates, gripper_name, out_dir / "grasp_grippers.ply", max_grasps)
        save_gripper_lines(candidates, gripper_name, out_dir / "grasp_lines.ply", max_grasps)
    except Exception as exc:
        logger.warning(f"Failed to write full grasp debug artifacts in {out_dir}: {exc}")
        return

    if render_enabled:
        render_pts = all_pts if show_scene else object_pts
        render_colors = all_colors if show_scene else object_colors
        render_pts, render_colors = sample_preview_points(render_pts, render_colors, render_max_points)
        render_debug_previews(
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


def write_result_json(out_dir: Path, result: dict) -> None:
    (out_dir / "grasp_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def failure_result_payload(
    *,
    text_prompt: str,
    confidence_threshold: float,
    grasp_threshold: float,
    output_mode: str,
    message: str,
    diagnostic_details: list[str],
    elapsed_ms: float = 0.0,
) -> dict:
    return {
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


def write_planner_result_json(
    *,
    out_dir: Path,
    text_prompt: str,
    confidence_threshold: float,
    grasp_threshold: float,
    elapsed_ms: float,
    output_mode: str,
    diagnostic: GraspDiagnostic,
    detection_confidence: float,
    detection_stamp_ns: int,
    depth_stamp_ns: int,
    depth_frame_id: str,
    camera_frame_id: str,
    camera_info_stamp_ns: int,
    depth_scale: float,
    camera_info_k: list[float],
    camera_info_width: int,
    camera_info_height: int,
    gripper_name: str,
    visualization_gripper: str,
    candidates,
    render_enabled: bool,
) -> dict:
    """The ROS-free body of GraspPlannerNode._save_debug_outputs' JSON write.

    Builds the planner result JSON (diagnostic mode and full mode) from plain
    values; the node supplies ROS message fields as scalars/strings.
    """
    result = {
        "prompt": text_prompt,
        "confidence_threshold": confidence_threshold,
        "grasp_threshold": grasp_threshold,
        "inference_time_ms": round(float(elapsed_ms), 3),
        "num_grasps": len(candidates),
        "debug_output_mode": output_mode,
        "diagnostic": diagnostic_to_dict(diagnostic, detection_confidence=detection_confidence),
        "depth_frame_id": depth_frame_id,
        "camera_frame_id": camera_frame_id,
        "detection_stamp_ns": int(detection_stamp_ns),
        "depth_stamp_ns": int(depth_stamp_ns),
        "camera_info_stamp_ns": int(camera_info_stamp_ns),
        "geometry_stamp_ns": int(depth_stamp_ns or detection_stamp_ns),
        "depth_detection_delta_ms": round(abs(depth_stamp_ns - detection_stamp_ns) / 1_000_000.0, 3),
        "depth_scale": depth_scale,
        "camera_info": {
            "k": list(camera_info_k),
            "width": camera_info_width,
            "height": camera_info_height,
        },
        "gripper": gripper_name,
        "visualization_gripper": visualization_gripper,
        "pointcloud": None,
        "render": {"scheduled": False},
        "grasps": [],
    }
    for i, g in enumerate(candidates):
        result["grasps"].append(candidate_debug_record(i, g))

    if output_mode == DEBUG_OUTPUT_DIAGNOSTIC:
        write_result_json(out_dir, result)
        return result

    object_raw_pts = debug_points_or_fallback(diagnostic.object_pc_raw, np.zeros((0, 3), dtype=np.float32))
    object_pts = debug_points_or_fallback(diagnostic.object_pc_after_completion, object_raw_pts)
    object_graspgen_input_pts = debug_points_or_fallback(diagnostic.object_pc_inference_input, object_pts)
    scene_pts = debug_points_or_fallback(diagnostic.scene_pc_after_completion, np.zeros((0, 3), dtype=np.float32))

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
    write_result_json(out_dir, result)
    return result


def start_full_debug_artifacts_thread(
    *,
    diagnostic: GraspDiagnostic,
    object_pts: np.ndarray,
    object_raw_pts: np.ndarray,
    object_graspgen_input_pts: np.ndarray,
    scene_pts: np.ndarray,
    candidates,
    vis_gripper: str,
    out_dir: Path,
    max_grasps: int,
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
    """The ROS-free thread launch of GraspPlannerNode._save_debug_outputs."""
    import threading

    object_colors = object_stage_colors(
        len(object_pts),
        diagnostic.object_point_count_raw,
        diagnostic.object_completion_added_count,
    )
    object_raw_colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (len(object_raw_pts), 1))
    object_graspgen_input_colors = np.tile(
        np.array([[0, 200, 255]], dtype=np.uint8),
        (len(object_graspgen_input_pts), 1),
    )
    scene_colors = scene_stage_colors(len(scene_pts), int(diagnostic.scene_point_count_raw) or len(scene_pts))
    thread = threading.Thread(
        target=write_full_debug_artifacts,
        kwargs={
            "object_pts": np.array(object_pts, dtype=np.float32, copy=True),
            "object_colors": np.array(object_colors, dtype=np.uint8, copy=True),
            "object_raw_pts": np.array(object_raw_pts, dtype=np.float32, copy=True),
            "object_raw_colors": np.array(object_raw_colors, dtype=np.uint8, copy=True),
            "object_graspgen_input_pts": np.array(object_graspgen_input_pts, dtype=np.float32, copy=True),
            "object_graspgen_input_colors": np.array(object_graspgen_input_colors, dtype=np.uint8, copy=True),
            "scene_pts": np.array(scene_pts, dtype=np.float32, copy=True),
            "scene_colors": np.array(scene_colors, dtype=np.uint8, copy=True),
            "candidates": candidates,
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
            "logger": logger,
        },
        name="grasp-debug-output",
        daemon=True,
    )
    thread.start()
    return thread
