#!/usr/bin/env python3
"""End-to-end wrist-camera target detection and pick test.

Pipeline:
  1. Move gripper to a configurable observation pose.
  2. Call GraspGen PlanGrasp for 6-DOF grasp candidates. GraspGen internally
     calls Grounded-SAM2 for the target mask.
  3. Convert each camera-frame GraspGen gripper pose to the robot base frame
     using current base->gripper TF and a hand-eye transform from either the
     runtime robot_config YAML or a calibration JSON report.
  4. Filter GraspGen candidates with workspace and IK checks.
  5. Optionally execute approach, descend, close gripper, and lift.

Legacy centroid mode is still available with --target-source centroid, but the
default path intentionally does not use the mask centroid as the grasp target.

This script intentionally lives outside ROS package entry points so it can be
edited quickly during hardware bring-up.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from ibrobot_msgs.action import ExecuteTaskPlan
from ibrobot_msgs.msg import Detection2D, TaskStep
from ibrobot_msgs.srv import DetectSegment, PlanGrasp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move to an observation pose, plan source grasps, align them to the configured target gripper, and optionally pick.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--prompt", default="banana", help="Text prompt passed to Grounded-SAM2/GraspGen")
    parser.add_argument(
        "--target-source",
        choices=("graspgen", "centroid"),
        default="graspgen",
        help="Target generator: GraspGen 6-DOF grasps by default; centroid is legacy fallback",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.10, help="Detection service confidence threshold"
    )
    parser.add_argument(
        "--min-confidence-for-pick",
        type=float,
        default=0.18,
        help="Centroid mode only: stop before pick if detection confidence is below this",
    )
    parser.add_argument(
        "--min-point-count",
        type=int,
        default=100,
        help="Centroid mode only: minimum valid depth points inside the mask",
    )
    parser.add_argument(
        "--manipulation-service",
        "--grasp-service",
        dest="manipulation_service",
        default="/grasp_planner/plan_grasp",
        help="PlanGrasp service name for GraspGen candidates",
    )
    parser.add_argument("--grasp-threshold", type=float, default=0.50, help="GraspGen discriminator threshold")
    parser.add_argument(
        "--debug-output-mode",
        choices=("default", "none", "diagnostic", "full"),
        default="default",
        help="Per-request PlanGrasp debug output: default follows the node parameter, diagnostic writes grasp_result.json, full also writes point clouds and previews",
    )
    parser.add_argument(
        "--min-grasp-confidence", type=float, default=0.0, help="Stop if a GraspGen candidate confidence is below this"
    )
    parser.add_argument(
        "--require-collision-free-grasp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require GraspGen candidates marked collision_free",
    )
    parser.add_argument(
        "--graspgen-rank-by-centroid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-rank similarly confident GraspGen candidates by contact-point distance to the detected 3D target centroid",
    )
    parser.add_argument(
        "--graspgen-centroid-confidence-window",
        type=float,
        default=0.06,
        help="Legacy compatibility knob; candidate scoring no longer hard-gates by confidence window",
    )
    parser.add_argument(
        "--graspgen-contact-distance-scale",
        type=float,
        default=0.06,
        help="Camera-frame contact-to-centroid distance where contact score reaches zero",
    )
    parser.add_argument(
        "--graspgen-topdown-weight",
        type=float,
        default=0.35,
        help="Weight for preferring top-down GraspGen candidates during re-ranking",
    )
    parser.add_argument(
        "--graspgen-topdown-min-z",
        type=float,
        default=-0.25,
        help="Minimum base-frame approach-axis z accepted by the top-down score",
    )
    parser.add_argument(
        "--graspgen-approach-distance",
        type=float,
        default=0.08,
        help="Pregrasp distance along GraspGen -Z approach axis",
    )
    parser.add_argument(
        "--graspgen-contact-x", type=float, default=0.0, help="GraspGen contact center x in GraspGen gripper frame"
    )
    parser.add_argument(
        "--graspgen-contact-y", type=float, default=0.0, help="GraspGen contact center y in GraspGen gripper frame"
    )
    parser.add_argument(
        "--graspgen-contact-z", type=float, default=0.195, help="GraspGen contact center z in GraspGen gripper frame"
    )
    parser.add_argument(
        "--target-contact-x",
        "--so101-contact-x",
        dest="target_contact_x",
        type=float,
        default=0.005,
        help="Target gripper effective contact center x in ee frame",
    )
    parser.add_argument(
        "--target-contact-y",
        "--so101-contact-y",
        dest="target_contact_y",
        type=float,
        default=0.0,
        help="Target gripper effective contact center y in ee frame",
    )
    parser.add_argument(
        "--target-contact-z",
        "--so101-contact-z",
        dest="target_contact_z",
        type=float,
        default=-0.075,
        help="Target gripper effective contact center z in ee frame",
    )
    parser.add_argument(
        "--target-auto-width-compensation",
        "--so101-auto-width-compensation",
        dest="target_auto_width_compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use candidate target_width_m to compensate asymmetric target-gripper contact center",
    )
    parser.add_argument(
        "--target-auto-width-required",
        "--so101-auto-width-required",
        dest="target_auto_width_required",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject GraspGen candidates without reliable target_width_m instead of falling back to static contact alignment",
    )
    parser.add_argument(
        "--target-width-quality-min",
        "--so101-width-quality-min",
        dest="target_width_quality_min",
        type=float,
        default=0.75,
        help="Minimum target_width_quality required for automatic target-gripper width compensation",
    )
    parser.add_argument(
        "--graspgen-to-ee-x",
        type=float,
        default=None,
        help="Static adapter x from source gripper frame to target ee frame; defaults to contact-center alignment",
    )
    parser.add_argument(
        "--graspgen-to-ee-y",
        type=float,
        default=None,
        help="Static adapter y from source gripper frame to target ee frame; defaults to contact-center alignment",
    )
    parser.add_argument(
        "--graspgen-to-ee-z",
        type=float,
        default=None,
        help="Static adapter z from source gripper frame to target ee frame; defaults to contact-center alignment",
    )
    parser.add_argument(
        "--graspgen-to-ee-roll",
        type=float,
        default=math.pi,
        help="Static adapter roll, radians, from source gripper frame to target ee frame",
    )
    parser.add_argument(
        "--graspgen-to-ee-pitch",
        type=float,
        default=0.0,
        help="Static adapter pitch, radians, from source gripper frame to target ee frame",
    )
    parser.add_argument(
        "--graspgen-to-ee-yaw",
        type=float,
        default=0.0,
        help="Static adapter yaw, radians, from source gripper frame to target ee frame",
    )
    parser.add_argument(
        "--contact-realign",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use actual target-gripper orientation to re-align the contact point before closing",
    )
    parser.add_argument(
        "--contact-realign-tolerance",
        type=float,
        default=0.010,
        help="Stop contact realignment when actual contact error is below this many meters",
    )
    parser.add_argument(
        "--contact-realign-max-iterations",
        type=int,
        default=4,
        help="Maximum correction moves after the first descent to reduce actual contact error",
    )

    parser.add_argument("--observe-x", type=float, default=-0.25, help="Observation gripper target x in base frame")
    parser.add_argument("--observe-y", type=float, default=-0.0, help="Observation gripper target y in base frame")
    parser.add_argument("--observe-z", type=float, default=0.20, help="Observation gripper target z in base frame")
    parser.add_argument(
        "--skip-observe", action="store_true", help="Do not move to observation pose; detect from current pose"
    )
    parser.add_argument(
        "--observe-only", action="store_true", help="Move to observation pose and exit before detection"
    )
    parser.add_argument(
        "--detect-only", action="store_true", help="Move/detect/compute targets but do not execute pick"
    )
    parser.add_argument(
        "--pick-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After key pick moves, log actual gripper pose and target/contact residuals",
    )
    parser.add_argument(
        "--pick-diagnostics-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-run target detection at the grasp pose to measure strawberry-to-gripper residual",
    )
    parser.add_argument(
        "--pick-diagnostics-settle-s",
        type=float,
        default=0.25,
        help="Settling time before sampling pick diagnostic TF/detection",
    )

    parser.add_argument(
        "--handeye-json",
        type=Path,
        default=Path("outputs/handeye/wrist_handeye.json"),
        help="Optional local hand-eye calibration JSON report used only with --handeye-source=json",
    )
    parser.add_argument(
        "--handeye-source",
        choices=("json", "robot-config"),
        default="robot-config",
        help="Source for the ee->camera transform",
    )
    parser.add_argument(
        "--handeye-key",
        choices=("ee_to_camera_optical", "ee_to_camera_link"),
        default="ee_to_camera_optical",
        help="Transform key in hand-eye JSON or frame convention for robot-config",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help="Runtime robot_config YAML used to launch the wrist camera; enables SSOT hand-eye and grasp_execution loading",
    )
    parser.add_argument("--camera-name", default="wrist", help="Camera peripheral name in --robot-config")
    parser.add_argument("--base-frame", default="base", help="Robot base frame for final target")
    parser.add_argument("--ee-frame", default="gripper", help="MoveIt end-effector frame")
    parser.add_argument(
        "--target-offset-x", type=float, default=0.0, help="Manual base-frame x correction added to the detected target"
    )
    parser.add_argument(
        "--target-offset-y", type=float, default=0.0, help="Manual base-frame y correction added to the detected target"
    )
    parser.add_argument(
        "--target-offset-z", type=float, default=0.0, help="Manual base-frame z correction added to the detected target"
    )
    parser.add_argument(
        "--handeye-warn-translation-std",
        type=float,
        default=0.02,
        help="Warn if any hand-eye validation translation std exceeds this many meters",
    )
    parser.add_argument(
        "--handeye-warn-rotation-rms-deg",
        type=float,
        default=5.0,
        help="Warn if hand-eye validation rotation RMS exceeds this many degrees",
    )
    parser.add_argument(
        "--handeye-warn-reprojection-px",
        type=float,
        default=5.0,
        help="Warn if hand-eye validation mean reprojection error exceeds this many pixels",
    )
    parser.add_argument(
        "--require-good-handeye",
        action="store_true",
        help="Abort before motion if hand-eye validation exceeds warning thresholds",
    )
    parser.add_argument(
        "--allow-handeye-config-mismatch",
        action="store_true",
        help="Continue even if --handeye-json disagrees with --robot-config",
    )
    parser.add_argument(
        "--handeye-config-mismatch-translation-m",
        type=float,
        default=0.02,
        help="Max allowed translation delta between hand-eye JSON and robot_config",
    )
    parser.add_argument(
        "--handeye-config-mismatch-rotation-deg",
        type=float,
        default=5.0,
        help="Max allowed rotation delta between hand-eye JSON and robot_config",
    )

    parser.add_argument("--tcp-x", type=float, default=0.005, help="Centroid mode only: target gripper effective TCP x")
    parser.add_argument("--tcp-y", type=float, default=0.0, help="Centroid mode only: target gripper effective TCP y")
    parser.add_argument(
        "--tcp-z", type=float, default=-0.105, help="Centroid mode only: target gripper effective TCP z"
    )
    parser.add_argument(
        "--tip-clearance-z",
        type=float,
        default=0.008,
        help="Centroid mode only: extra vertical clearance added to the gripper target",
    )
    parser.add_argument(
        "--approach-lift",
        type=float,
        default=0.115,
        help="Centroid mode only: approach height above computed grasp target",
    )
    parser.add_argument("--final-lift", type=float, default=0.165, help="Lift height above computed grasp target")

    parser.add_argument(
        "--sample-xy-radius",
        type=float,
        default=0.04,
        help="XY radius for candidate grasp sampling around the detected target",
    )
    parser.add_argument("--sample-xy-step", type=float, default=0.02, help="XY grid step for candidate grasp sampling")
    parser.add_argument(
        "--sample-z-offsets",
        default="0,0.02,0.04,0.06,-0.01",
        help="Comma-separated z offsets added to the compensated grasp target",
    )
    parser.add_argument("--max-candidates", type=int, default=80, help="Maximum sampled candidates to test")

    parser.add_argument(
        "--max-target-radius", type=float, default=0.52, help="Workspace guard: max radius of compensated grasp target"
    )
    parser.add_argument(
        "--max-abs-y", type=float, default=0.45, help="Workspace guard: max absolute y of compensated grasp target"
    )
    parser.add_argument(
        "--min-grasp-z",
        type=float,
        default=0.02,
        help="Workspace guard: minimum base-frame z for the commanded target ee grasp pose",
    )
    parser.add_argument(
        "--min-contact-z",
        type=float,
        default=0.0,
        help="Workspace guard: minimum base-frame z for the aligned contact point",
    )
    parser.add_argument(
        "--min-approach-z",
        type=float,
        default=0.04,
        help="Workspace guard: minimum base-frame z for the pregrasp approach pose",
    )
    parser.add_argument(
        "--allow-out-of-workspace", action="store_true", help="Disable workspace guard and send the pick anyway"
    )

    parser.add_argument(
        "--ik-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter sampled candidates with MoveIt compute_ik before executing",
    )
    parser.add_argument("--ik-service", default="/compute_ik", help="MoveIt GetPositionIK service name")
    parser.add_argument("--ik-group", default="arm", help="MoveIt group name used for IK filtering")
    parser.add_argument("--ik-timeout-s", type=float, default=0.20, help="Per-candidate IK timeout inside MoveIt")
    parser.add_argument(
        "--ik-wait-timeout-s", type=float, default=12.0, help="Timeout while waiting for the IK service"
    )
    parser.add_argument("--ik-avoid-collisions", action="store_true", help="Ask compute_ik to avoid collisions")
    parser.add_argument(
        "--ik-check-orientation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check candidate orientation in /compute_ik; default filters by position for underactuated arms",
    )
    parser.add_argument(
        "--execute-grasp-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send GraspGen orientation to MoveIt execution so contact compensation matches the commanded gripper pose",
    )
    parser.add_argument(
        "--require-grasp-ik",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the final grasp pose to pass IK",
    )
    parser.add_argument(
        "--require-lift-ik",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the lift pose to pass IK",
    )

    parser.add_argument("--observe-speed", type=float, default=0.04, help="Velocity scaling for observation move")
    parser.add_argument("--approach-speed", type=float, default=0.05, help="Velocity scaling for approach move")
    parser.add_argument("--descend-speed", type=float, default=0.03, help="Velocity scaling for descend move")
    parser.add_argument("--lift-speed", type=float, default=0.05, help="Velocity scaling for lift move")
    parser.add_argument("--observe-settle-s", type=float, default=0.6, help="Wait after reaching observation pose")
    parser.add_argument("--open-settle-s", type=float, default=0.3, help="Wait after opening gripper")
    parser.add_argument("--hold-s", type=float, default=0.8, help="Wait after closing gripper before lift")

    parser.add_argument(
        "--detect-service", default="/grounded_sam2/detect_and_segment", help="DetectSegment service name"
    )
    parser.add_argument("--task-action", default="/task_executor/execute_task_plan", help="ExecuteTaskPlan action name")
    parser.add_argument("--ready-timeout-s", type=float, default=12.0, help="Service/action wait timeout")
    parser.add_argument("--detect-timeout-s", type=float, default=90.0, help="Detection call timeout")
    parser.add_argument("--task-timeout-s", type=float, default=150.0, help="Task execution timeout")
    parser.add_argument("--tf-timeout-s", type=float, default=10.0, help="TF lookup timeout")

    args = parser.parse_args()
    args._graspgen_to_ee_translation_auto = (  # noqa: SLF001 - script-local diagnostic state.
        args.graspgen_to_ee_x is None and args.graspgen_to_ee_y is None and args.graspgen_to_ee_z is None
    )
    load_grasp_execution_config(args)
    resolve_graspgen_adapter_defaults(args)
    return args


def _float_triplet(value, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-element list")
    return (float(value[0]), float(value[1]), float(value[2]))


def _normalized_vector(value: Iterable[float], name: str) -> np.ndarray:
    vector = np.array(list(value), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def load_grasp_execution_config(args: argparse.Namespace) -> None:
    args.grasp_execution_config = {}
    args.target_fixed_finger_contact_ee = None
    args.target_closing_axis_ee = None
    args.target_width_clearance_m = 0.003
    args.target_width_min_m = 0.005
    args.target_width_max_m = 0.08
    args.target_width_fallback_m = 0.035
    args.execution_scoring_config = {}
    args.execution_contact_distance_weight = 1.0
    args.execution_topdown_weight = float(args.graspgen_topdown_weight)
    args.execution_confidence_weight = 1.0
    args.execution_contact_distance_scale = float(args.graspgen_contact_distance_scale)

    if args.robot_config is None or not args.robot_config.exists():
        return

    payload = yaml.safe_load(args.robot_config.read_text(encoding="utf-8")) or {}
    robot = payload.get("robot") if isinstance(payload, dict) else None
    if not isinstance(robot, dict):
        return

    config = robot.get("grasp_execution")
    if not isinstance(config, dict):
        return

    target_gripper = config.get("target_gripper", {})
    if not isinstance(target_gripper, dict):
        target_gripper = {}

    fixed = target_gripper.get("fixed_finger_contact_ee")
    axis = target_gripper.get("closing_axis_ee")
    if fixed is not None:
        args.target_fixed_finger_contact_ee = _float_triplet(
            fixed, "grasp_execution.target_gripper.fixed_finger_contact_ee"
        )
    if axis is not None:
        args.target_closing_axis_ee = tuple(
            float(v) for v in _normalized_vector(axis, "grasp_execution.target_gripper.closing_axis_ee")
        )

    if "width_clearance_m" in target_gripper:
        args.target_width_clearance_m = float(target_gripper["width_clearance_m"])
    if "min_width_m" in target_gripper:
        args.target_width_min_m = float(target_gripper["min_width_m"])
    if "max_width_m" in target_gripper:
        args.target_width_max_m = float(target_gripper["max_width_m"])
    if "fallback_width_m" in target_gripper:
        args.target_width_fallback_m = float(target_gripper["fallback_width_m"])
    if "width_quality_min" in target_gripper:
        args.target_width_quality_min = float(target_gripper["width_quality_min"])

    scoring = config.get("execution_scoring", {})
    if isinstance(scoring, dict):
        args.execution_scoring_config = scoring
        if "contact_distance_weight" in scoring:
            args.execution_contact_distance_weight = float(scoring["contact_distance_weight"])
        if "topdown_weight" in scoring:
            args.execution_topdown_weight = float(scoring["topdown_weight"])
        if "confidence_weight" in scoring:
            args.execution_confidence_weight = float(scoring["confidence_weight"])
        if "contact_distance_scale_m" in scoring:
            args.execution_contact_distance_scale = float(scoring["contact_distance_scale_m"])

    source_contact = config.get("source_contact_point")
    if source_contact is not None:
        args.graspgen_contact_x, args.graspgen_contact_y, args.graspgen_contact_z = _float_triplet(
            source_contact, "grasp_execution.source_contact_point"
        )

    adapter = config.get("adapter", {})
    if isinstance(adapter, dict):
        rpy = adapter.get("source_to_ee_rpy")
        if rpy is not None:
            args.graspgen_to_ee_roll, args.graspgen_to_ee_pitch, args.graspgen_to_ee_yaw = _float_triplet(
                rpy, "grasp_execution.adapter.source_to_ee_rpy"
            )

    args.grasp_execution_config = config


def _static_target_contact(args: argparse.Namespace) -> np.ndarray:
    return np.array([args.target_contact_x, args.target_contact_y, args.target_contact_z], dtype=np.float64)


def _adapter_translation(
    args: argparse.Namespace,
    target_contact: np.ndarray,
) -> tuple[float, float, float]:
    p_graspgen = np.array(
        [args.graspgen_contact_x, args.graspgen_contact_y, args.graspgen_contact_z],
        dtype=np.float64,
    )
    rotation = R.from_euler(
        "xyz",
        [args.graspgen_to_ee_roll, args.graspgen_to_ee_pitch, args.graspgen_to_ee_yaw],
    ).as_matrix()
    translation = p_graspgen - rotation @ target_contact
    return (float(translation[0]), float(translation[1]), float(translation[2]))


def candidate_target_width(candidate) -> tuple[float, float]:
    width = float(getattr(candidate, "target_width_m", 0.0))
    quality = float(getattr(candidate, "target_width_quality", 0.0))
    if not math.isfinite(width):
        width = 0.0
    if not math.isfinite(quality):
        quality = 0.0
    return width, quality


def resolve_target_contact_for_candidate(args: argparse.Namespace, candidate) -> tuple[np.ndarray, str]:
    static_contact = _static_target_contact(args)
    if not args.target_auto_width_compensation:
        return static_contact, "static:auto_disabled"
    if not args._graspgen_to_ee_translation_auto:
        return static_contact, "static:manual_adapter"
    if args.target_fixed_finger_contact_ee is None or args.target_closing_axis_ee is None:
        return static_contact, "static:missing_target_gripper_config"

    measured_width, quality = candidate_target_width(candidate)
    if measured_width <= 0.0 or quality < float(args.target_width_quality_min):
        if args.target_auto_width_required:
            raise ValueError(f"target_width_unreliable width={measured_width:.4f} quality={quality:.3f}")
        width = float(args.target_width_fallback_m)
        source = "fallback"
    else:
        width = measured_width
        source = "auto"

    width = min(max(width, float(args.target_width_min_m)), float(args.target_width_max_m))
    width_with_clearance = min(
        max(width + float(args.target_width_clearance_m), float(args.target_width_min_m)),
        float(args.target_width_max_m),
    )
    fixed_contact = np.array(args.target_fixed_finger_contact_ee, dtype=np.float64)
    width_axis = np.array(args.target_closing_axis_ee, dtype=np.float64)
    contact = fixed_contact + width_axis * (0.5 * width_with_clearance)
    reason = (
        f"{source}:measured_width={measured_width:.4f}:used_width={width:.4f}:quality={quality:.3f}:"
        f"width_with_clearance={width_with_clearance:.4f}"
    )
    return contact, reason


def resolve_graspgen_adapter_defaults(args: argparse.Namespace) -> None:
    translation = _adapter_translation(args, _static_target_contact(args))

    if args.graspgen_to_ee_x is None:
        args.graspgen_to_ee_x = translation[0]
    if args.graspgen_to_ee_y is None:
        args.graspgen_to_ee_y = translation[1]
    if args.graspgen_to_ee_z is None:
        args.graspgen_to_ee_z = translation[2]


def parse_float_csv(text: str) -> list[float]:
    values: list[float] = []
    for raw in text.split(","):
        item = raw.strip()
        if item:
            values.append(float(item))
    return values or [0.0]


def quat_rotate(
    q_xyzw: tuple[float, float, float, float], v_xyz: tuple[float, float, float]
) -> tuple[float, float, float]:
    qx, qy, qz, qw = q_xyzw
    vx, vy, vz = v_xyz
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        return v_xyz
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def mat4_mul_point(matrix: list[list[float]], point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def make_pose(x: float, y: float, z: float, quat_xyzw: tuple[float, float, float, float] | None = None) -> Pose:
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    qx, qy, qz, qw = quat_xyzw or (0.0, 0.0, 0.0, 1.0)
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)
    return pose


def make_move_step(
    label: str,
    xyz: tuple[float, float, float],
    speed: float,
    quat_xyzw: tuple[float, float, float, float] | None = None,
) -> TaskStep:
    step = TaskStep()
    step.type = TaskStep.MOVE_TO_POSE
    step.label = label
    step.target_pose = make_pose(*xyz, quat_xyzw)
    step.velocity_scaling = float(speed)
    return step


def make_gripper_step(label: str, position: float) -> TaskStep:
    step = TaskStep()
    step.type = TaskStep.GRIPPER
    step.label = label
    step.gripper_position = float(position)
    return step


def make_wait_step(label: str, seconds: float) -> TaskStep:
    step = TaskStep()
    step.type = TaskStep.WAIT
    step.label = label
    step.wait_duration_s = float(seconds)
    return step


def fmt_xyz(xyz: Iterable[float]) -> str:
    x, y, z = xyz
    return f"({x:.4f},{y:.4f},{z:.4f})"


def fmt_quat(q_xyzw: Iterable[float]) -> str:
    qx, qy, qz, qw = q_xyzw
    return f"({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f})"


def transform_to_matrix(transform) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def adapter_matrix(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    matrix[:3, 3] = [x, y, z]
    return matrix


def camera_link_to_optical_matrix() -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R.from_quat([-0.5, 0.5, -0.5, 0.5]).as_matrix()
    return matrix


def pose_from_matrix(matrix: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    xyz = tuple(float(v) for v in matrix[:3, 3])
    quat = tuple(float(v) for v in R.from_matrix(matrix[:3, :3]).as_quat())
    return xyz, quat


def matrix_from_pose(
    xyz: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float] | None
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R.from_quat(quat_xyzw or (0.0, 0.0, 0.0, 1.0)).as_matrix()
    matrix[:3, 3] = [xyz[0], xyz[1], xyz[2]]
    return matrix


def matrix_from_rowmajor(values: Iterable[float]) -> np.ndarray:
    matrix = np.array(list(values), dtype=np.float64).reshape(4, 4)
    matrix[3, :] = [0.0, 0.0, 0.0, 1.0]
    return matrix


def add_xyz(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def sub_xyz(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def norm_xyz(xyz: tuple[float, float, float]) -> float:
    return math.sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2])


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def quat_delta_deg(
    actual_xyzw: tuple[float, float, float, float],
    commanded_xyzw: tuple[float, float, float, float],
) -> float:
    delta = R.from_quat(commanded_xyzw).inv() * R.from_quat(actual_xyzw)
    return math.degrees(float(delta.magnitude()))


def _load_robot_camera_transform(config_path: Path, camera_name: str, ee_frame: str) -> tuple[np.ndarray, dict]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    robot = payload.get("robot")
    if not isinstance(robot, dict):
        raise ValueError(f"{config_path} does not contain robot config")

    cameras = [
        item for item in robot.get("peripherals", []) or [] if isinstance(item, dict) and item.get("type") == "camera"
    ]
    camera = next((item for item in cameras if item.get("name") == camera_name), None)
    if camera is None:
        names = ", ".join(str(item.get("name", "")) for item in cameras) or "none"
        raise KeyError(f"camera {camera_name!r} not found in {config_path}; configured cameras: {names}")

    transform = camera.get("transform")
    if not isinstance(transform, dict):
        raise ValueError(f"camera {camera_name!r} in {config_path} has no transform")

    parent = str(transform.get("parent_frame", ""))
    if parent != ee_frame:
        raise ValueError(
            f"camera {camera_name!r} transform parent_frame={parent!r} does not match ee_frame={ee_frame!r}"
        )

    required = ("x", "y", "z", "roll", "pitch", "yaw")
    missing = [key for key in required if key not in transform]
    if missing:
        raise ValueError(f"camera {camera_name!r} transform is missing keys: {', '.join(missing)}")

    matrix = adapter_matrix(
        float(transform["x"]),
        float(transform["y"]),
        float(transform["z"]),
        float(transform["roll"]),
        float(transform["pitch"]),
        float(transform["yaw"]),
    )
    return matrix, camera


def _rotation_delta_deg(left: np.ndarray, right: np.ndarray) -> float:
    delta = R.from_matrix(left[:3, :3]).inv() * R.from_matrix(right[:3, :3])
    return math.degrees(float(delta.magnitude()))


class BananaHandeyePickClient(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("banana_handeye_pick_test")
        self.args = args
        self.detect_client = self.create_client(DetectSegment, args.detect_service)
        self.grasp_client = self.create_client(PlanGrasp, args.manipulation_service)
        self.task_client = ActionClient(self, ExecuteTaskPlan, args.task_action)
        self.ik_client = self.create_client(GetPositionIK, args.ik_service) if args.ik_filter else None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.handeye_data, self.handeye_matrix = self._load_handeye(args)
        self.selected_target_contact_ee: tuple[float, float, float] | None = None
        self.selected_plan_contact_base: tuple[float, float, float] | None = None
        self.observed_target_base: tuple[float, float, float] | None = None
        self.current_plan_contact_base: tuple[float, float, float] | None = None

    @staticmethod
    def _load_handeye_data(path: Path) -> dict:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a hand-eye JSON object")
        return data

    @staticmethod
    def _extract_handeye_matrix(data: dict, path: Path, key: str) -> list[list[float]]:
        if key not in data or "matrix" not in data[key]:
            raise KeyError(f"{path} does not contain {key}.matrix")
        return data[key]["matrix"]

    @classmethod
    def _load_handeye(cls, args: argparse.Namespace) -> tuple[dict, list[list[float]]]:
        if args.handeye_source == "robot-config":
            if args.robot_config is None:
                raise ValueError("--robot-config is required when --handeye-source=robot-config")
            t_ee_camera_link, camera = _load_robot_camera_transform(args.robot_config, args.camera_name, args.ee_frame)
            matrix = (
                t_ee_camera_link
                if args.handeye_key == "ee_to_camera_link"
                else t_ee_camera_link @ camera_link_to_optical_matrix()
            )
            data = {
                "source": "robot-config",
                "robot_config": str(args.robot_config),
                "camera_name": args.camera_name,
                "camera_frame_id": camera.get("frame_id", ""),
                "camera_optical_frame_id": camera.get("optical_frame_id", ""),
            }
            print(
                f"HANDEYE_SOURCE source=robot-config config={args.robot_config} camera={args.camera_name} "
                f"key={args.handeye_key}",
                flush=True,
            )
            return data, matrix.tolist()

        data = cls._load_handeye_data(args.handeye_json)
        matrix = cls._extract_handeye_matrix(data, args.handeye_json, args.handeye_key)
        print(f"HANDEYE_SOURCE source=json path={args.handeye_json} key={args.handeye_key}", flush=True)
        if args.robot_config is not None:
            cls._check_handeye_matches_robot_config(data, args)
        return data, matrix

    @classmethod
    def _check_handeye_matches_robot_config(cls, data: dict, args: argparse.Namespace) -> None:
        json_link = data.get("ee_to_camera_link", {}).get("matrix")
        if json_link is None:
            print("HANDEYE_CONFIG_CHECK skipped=True reason=json_missing_ee_to_camera_link", flush=True)
            return

        t_json = np.array(json_link, dtype=np.float64)
        t_config, _ = _load_robot_camera_transform(args.robot_config, args.camera_name, args.ee_frame)
        translation_delta = float(np.linalg.norm(t_config[:3, 3] - t_json[:3, 3]))
        rotation_delta = _rotation_delta_deg(t_config, t_json)
        print(
            f"HANDEYE_CONFIG_CHECK config={args.robot_config} camera={args.camera_name} "
            f"translation_delta_m={translation_delta:.4f} rotation_delta_deg={rotation_delta:.2f}",
            flush=True,
        )
        bad = translation_delta > float(args.handeye_config_mismatch_translation_m) or rotation_delta > float(
            args.handeye_config_mismatch_rotation_deg
        )
        if bad and not args.allow_handeye_config_mismatch:
            raise RuntimeError(
                "Hand-eye JSON and robot_config camera transform disagree: "
                f"translation_delta={translation_delta:.4f}m "
                f"(limit {args.handeye_config_mismatch_translation_m:.4f}m), "
                f"rotation_delta={rotation_delta:.2f}deg "
                f"(limit {args.handeye_config_mismatch_rotation_deg:.2f}deg). "
                "Use --handeye-source robot-config or update one source."
            )

    def check_handeye_quality(self) -> None:
        validation = self.handeye_data.get("validation")
        if not isinstance(validation, dict):
            source = self.handeye_data.get("source", "unknown")
            print(f"HANDEYE_QUALITY status=unknown source={source} reason=no_validation_metrics", flush=True)
            return

        translation_std = validation.get("target_translation_std_m", [])
        if not isinstance(translation_std, list):
            translation_std = []
        max_translation_std = max((abs(float(item)) for item in translation_std), default=0.0)
        rotation_rms = float(validation.get("target_rotation_rms_deg", 0.0))
        reprojection = float(validation.get("reprojection_mean_px", 0.0))

        bad_reasons: list[str] = []
        if max_translation_std > self.args.handeye_warn_translation_std:
            bad_reasons.append(
                f"translation_std_max={max_translation_std:.4f}>{self.args.handeye_warn_translation_std:.4f}m"
            )
        if rotation_rms > self.args.handeye_warn_rotation_rms_deg:
            bad_reasons.append(f"rotation_rms={rotation_rms:.2f}>{self.args.handeye_warn_rotation_rms_deg:.2f}deg")
        if reprojection > self.args.handeye_warn_reprojection_px:
            bad_reasons.append(f"reprojection={reprojection:.2f}>{self.args.handeye_warn_reprojection_px:.2f}px")

        status = "bad" if bad_reasons else "ok"
        print(
            f"HANDEYE_QUALITY status={status} translation_std_max={max_translation_std:.4f} "
            f"rotation_rms_deg={rotation_rms:.2f} reprojection_px={reprojection:.2f}",
            flush=True,
        )
        if bad_reasons:
            print(f"HANDEYE_WARNING reasons={';'.join(bad_reasons)}", flush=True)
            if self.args.require_good_handeye:
                raise RuntimeError("Hand-eye validation is outside thresholds")

    def wait_ready(self) -> None:
        needs_perception = not self.args.observe_only
        needs_detection = self.args.target_source == "centroid" or (
            self.args.target_source == "graspgen" and self.args.graspgen_rank_by_centroid
        )
        if (
            needs_perception
            and self.args.target_source == "graspgen"
            and not self.grasp_client.wait_for_service(timeout_sec=self.args.ready_timeout_s)
        ):
            raise RuntimeError(f"Manipulation service is not available: {self.args.manipulation_service}")
        if (
            needs_perception
            and needs_detection
            and not self.detect_client.wait_for_service(timeout_sec=self.args.ready_timeout_s)
        ):
            raise RuntimeError(f"Detect service is not available: {self.args.detect_service}")
        needs_task_action = not self.args.skip_observe or not self.args.detect_only
        if needs_task_action and not self.task_client.wait_for_server(timeout_sec=self.args.ready_timeout_s):
            raise RuntimeError(f"Task action is not available: {self.args.task_action}")

    def wait_ik_ready(self) -> None:
        if (
            self.args.ik_filter
            and self.ik_client is not None
            and not self.ik_client.wait_for_service(timeout_sec=self.args.ik_wait_timeout_s)
        ):
            raise RuntimeError(f"IK service is not available: {self.args.ik_service}")

    def run_task(self, task_id: str, description: str, steps: list[TaskStep], timeout_s: float | None = None) -> bool:
        goal = ExecuteTaskPlan.Goal()
        goal.task_id = task_id
        goal.task_description = description
        goal.steps = steps

        print(f"TASK_SEND id={task_id} steps={len(steps)} desc={description}", flush=True)
        future = self.task_client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done():
            raise RuntimeError(f"Timed out sending task: {task_id}")
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"Task was rejected: {task_id}")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_s or self.args.task_timeout_s)
        if not result_future.done():
            raise RuntimeError(f"Timed out waiting for task result: {task_id}")

        result = result_future.result().result
        print(
            f"TASK_RESULT success={result.success} steps={result.steps_completed} "
            f"duration={result.total_duration_s:.2f}s msg={result.message}",
            flush=True,
        )
        return bool(result.success)

    @staticmethod
    def _feedback_cb(feedback_msg) -> None:
        feedback = feedback_msg.feedback
        print(
            f"TASK_FEEDBACK step={feedback.current_step}/{feedback.total_steps} "
            f"label={feedback.current_label} status={feedback.status}",
            flush=True,
        )

    def move_to_observe(self) -> None:
        if self.args.skip_observe:
            print("OBSERVE skipped=True", flush=True)
            return

        observe_xyz = (self.args.observe_x, self.args.observe_y, self.args.observe_z)
        steps = [
            make_move_step("move_to_observation_pose", observe_xyz, self.args.observe_speed),
            make_wait_step("settle_observation_image", self.args.observe_settle_s),
        ]
        ok = self.run_task(
            "banana_observe_pose", f"move to observation pose {fmt_xyz(observe_xyz)}", steps, timeout_s=90.0
        )
        if not ok:
            raise RuntimeError("Failed to move to observation pose")

    def lookup_base_to_gripper(self):
        deadline = time.monotonic() + self.args.tf_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                transform = self.tf_buffer.lookup_transform(self.args.base_frame, self.args.ee_frame, Time())
                t = transform.transform.translation
                q = transform.transform.rotation
                print(
                    f"GRIPPER_BASE x={t.x:.4f} y={t.y:.4f} z={t.z:.4f} q=({q.x:.4f},{q.y:.4f},{q.z:.4f},{q.w:.4f})",
                    flush=True,
                )
                return transform
            except Exception as exc:  # tf2 raises several exception classes.
                last_error = exc
        raise RuntimeError(f"No TF from {self.args.ee_frame} to {self.args.base_frame}: {last_error}")

    def detect_target(self) -> Detection2D:
        request = DetectSegment.Request()
        request.text_prompt = self.args.prompt
        request.confidence_threshold = self.args.confidence_threshold

        print(f"DETECT_SEND prompt={self.args.prompt} threshold={self.args.confidence_threshold}", flush=True)
        future = self.detect_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.detect_timeout_s)
        if not future.done():
            raise RuntimeError("Detection timed out")

        response = future.result()
        if not response.success:
            raise RuntimeError(f"Detection failed: {response.message}")

        candidates = [
            item
            for item in response.detections.detections
            if self.args.prompt.lower() in item.label.lower() and item.point_count >= self.args.min_point_count
        ]
        if not candidates:
            raise RuntimeError(
                f"No valid {self.args.prompt!r} detection; detections={len(response.detections.detections)}"
            )

        detection = max(candidates, key=lambda item: (float(item.confidence), int(item.point_count)))
        print(
            f"DETECTION label={detection.label} conf={float(detection.confidence):.3f} "
            f"points={int(detection.point_count)} frame={detection.header.frame_id} "
            f"camera_xyz=({detection.centroid_xyz.x:.4f},{detection.centroid_xyz.y:.4f},{detection.centroid_xyz.z:.4f})",
            flush=True,
        )
        return detection

    def request_graspgen_candidates(self):
        request = PlanGrasp.Request()
        request.text_prompt = self.args.prompt
        request.confidence_threshold = self.args.confidence_threshold
        request.grasp_threshold = self.args.grasp_threshold
        request.debug_output_mode = self.args.debug_output_mode

        print(
            f"GRASPGEN_SEND prompt={self.args.prompt} detect_threshold={self.args.confidence_threshold} "
            f"grasp_threshold={self.args.grasp_threshold} debug_output_mode={self.args.debug_output_mode}",
            flush=True,
        )
        future = self.grasp_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.detect_timeout_s)
        if not future.done():
            raise RuntimeError("GraspGen planning timed out")

        response = future.result()
        self.print_graspgen_diagnostics(response)
        if not response.success:
            raise RuntimeError(f"GraspGen planning failed: {response.message}")

        candidates = list(response.grasps.grasps)
        print(
            f"GRASPGEN_RESULT success=True n={len(candidates)} inference_ms={float(response.inference_time_ms):.1f} "
            f"frame={response.grasps.header.frame_id} msg={response.message}",
            flush=True,
        )
        if not candidates:
            raise RuntimeError("GraspGen returned zero candidates")
        return candidates

    @staticmethod
    def print_graspgen_diagnostics(response) -> None:
        debug_output_dir = getattr(response, "debug_output_dir", "")
        if debug_output_dir:
            print(f"GRASPGEN_DEBUG_OUTPUT dir={debug_output_dir}", flush=True)
        for item in getattr(response, "diagnostic_details", []):
            print(f"GRASPGEN_DIAGNOSTIC {item}", flush=True)

    def graspgen_to_base_pose(self, candidate, base_to_gripper_tf):
        t_base_gripper = transform_to_matrix(base_to_gripper_tf)
        t_gripper_camera = np.array(self.handeye_matrix, dtype=np.float64)
        t_camera_graspgen = matrix_from_rowmajor(candidate.pose_matrix)
        target_contact, width_reason = resolve_target_contact_for_candidate(self.args, candidate)
        adapter_xyz = (
            _adapter_translation(self.args, target_contact)
            if self.args._graspgen_to_ee_translation_auto
            else (self.args.graspgen_to_ee_x, self.args.graspgen_to_ee_y, self.args.graspgen_to_ee_z)
        )
        t_graspgen_ee = adapter_matrix(
            adapter_xyz[0],
            adapter_xyz[1],
            adapter_xyz[2],
            self.args.graspgen_to_ee_roll,
            self.args.graspgen_to_ee_pitch,
            self.args.graspgen_to_ee_yaw,
        )
        t_base_graspgen = t_base_gripper @ t_gripper_camera @ t_camera_graspgen
        t_base_ee = t_base_graspgen @ t_graspgen_ee
        return t_base_ee, t_base_graspgen, t_camera_graspgen, target_contact, adapter_xyz, width_reason

    def graspgen_targets_from_pose(
        self, t_base_ee: np.ndarray, t_base_graspgen: np.ndarray
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float, float],
        float,
    ]:
        grasp, quat = pose_from_matrix(t_base_ee)
        offset = (self.args.target_offset_x, self.args.target_offset_y, self.args.target_offset_z)
        if offset != (0.0, 0.0, 0.0):
            grasp = add_xyz(grasp, offset)
            print(f"TARGET_OFFSET base_offset={fmt_xyz(offset)} corrected_grasp={fmt_xyz(grasp)}", flush=True)

        approach_axis = tuple(float(v) for v in t_base_graspgen[:3, 2])
        approach = (
            grasp[0] - approach_axis[0] * self.args.graspgen_approach_distance,
            grasp[1] - approach_axis[1] * self.args.graspgen_approach_distance,
            grasp[2] - approach_axis[2] * self.args.graspgen_approach_distance,
        )
        lift = (grasp[0], grasp[1], grasp[2] + self.args.final_lift)
        radius = math.sqrt(grasp[0] * grasp[0] + grasp[1] * grasp[1] + grasp[2] * grasp[2])
        return approach, grasp, lift, quat, radius

    def graspgen_contact_point_base(self, t_base_graspgen: np.ndarray) -> tuple[float, float, float]:
        contact = t_base_graspgen @ np.array(
            [
                self.args.graspgen_contact_x,
                self.args.graspgen_contact_y,
                self.args.graspgen_contact_z,
                1.0,
            ],
            dtype=np.float64,
        )
        return (float(contact[0]), float(contact[1]), float(contact[2]))

    def graspgen_contact_point_camera(self, t_camera_graspgen: np.ndarray) -> tuple[float, float, float]:
        contact = t_camera_graspgen @ np.array(
            [
                self.args.graspgen_contact_x,
                self.args.graspgen_contact_y,
                self.args.graspgen_contact_z,
                1.0,
            ],
            dtype=np.float64,
        )
        return (float(contact[0]), float(contact[1]), float(contact[2]))

    def rank_graspgen_candidates(self, candidates, base_to_gripper_tf) -> list[tuple[int, object, float | None, float]]:
        indexed = [(index, candidate, None, 0.0) for index, candidate in enumerate(candidates)]
        topdown_weight = max(0.0, float(self.args.execution_topdown_weight))
        contact_weight = max(0.0, float(self.args.execution_contact_distance_weight))
        confidence_weight = max(0.0, float(self.args.execution_confidence_weight))
        if not self.args.graspgen_rank_by_centroid and topdown_weight <= 0.0 and contact_weight <= 0.0:
            print("GRASPGEN_RANK enabled=False", flush=True)
            return indexed

        centroid = None
        centroid_reason = "disabled"
        try:
            if self.args.graspgen_rank_by_centroid:
                detection = self.detect_target()
                centroid = np.array(
                    [
                        detection.centroid_xyz.x,
                        detection.centroid_xyz.y,
                        detection.centroid_xyz.z,
                    ],
                    dtype=np.float64,
                )
                self.observed_target_base = self.detection_to_base(detection, base_to_gripper_tf)
                centroid_reason = "ok"
        except Exception as exc:
            centroid_reason = str(exc)

        legacy_window = max(0.0, float(self.args.graspgen_centroid_confidence_window))
        contact_distance_scale = max(1e-6, float(self.args.execution_contact_distance_scale))
        min_topdown_z = float(self.args.graspgen_topdown_min_z)
        min_topdown_dot = max(0.0, min(0.999, -min_topdown_z))
        t_base_gripper = transform_to_matrix(base_to_gripper_tf)
        t_gripper_camera = np.array(self.handeye_matrix, dtype=np.float64)
        scored: list[tuple[int, object, float, float | None, float, float]] = []
        for index, candidate in enumerate(candidates):
            t_camera_graspgen = matrix_from_rowmajor(candidate.pose_matrix)
            distance = None
            if centroid is not None:
                contact = self.graspgen_contact_point_camera(t_camera_graspgen)
                distance = float(np.linalg.norm(np.array(contact, dtype=np.float64) - centroid))
            t_base_graspgen = t_base_gripper @ t_gripper_camera @ t_camera_graspgen
            approach_axis_base = tuple(float(v) for v in t_base_graspgen[:3, 2])
            topdown_dot = max(0.0, -approach_axis_base[2])
            topdown_score = max(0.0, min(1.0, (topdown_dot - min_topdown_dot) / (1.0 - min_topdown_dot)))
            confidence = float(candidate.confidence)
            contact_score = 0.0 if distance is None else 1.0 - clamp(distance / contact_distance_scale, 0.0, 1.0)
            combined = confidence_weight * confidence + topdown_weight * topdown_score + contact_weight * contact_score
            scored.append((index, candidate, confidence, distance, topdown_score, combined))

        scored.sort(key=lambda item: (-item[5], float("inf") if item[3] is None else item[3], -item[2], item[0]))
        summary = ",".join(
            f"{index}:{'nan' if distance is None else f'{distance:.4f}'}/{confidence:.3f}/td={topdown:.3f}/s={combined:.3f}"
            for index, _, confidence, distance, topdown, combined in scored[:10]
        )
        centroid_text = (
            fmt_xyz(tuple(float(v) for v in centroid)) if centroid is not None else f"unavailable:{centroid_reason}"
        )
        print(
            f"GRASPGEN_RANK enabled=True centroid_camera={centroid_text} "
            f"legacy_confidence_window={legacy_window:.3f} contact_distance_scale={contact_distance_scale:.3f} "
            f"confidence_weight={confidence_weight:.3f} "
            f"contact_weight={contact_weight:.3f} topdown_weight={topdown_weight:.3f} "
            f"topdown_min_z={min_topdown_z:.3f} order={summary}",
            flush=True,
        )
        return [(index, candidate, distance, topdown) for index, candidate, _, distance, topdown, _ in scored]

    def select_graspgen_candidate(
        self,
        base_to_gripper_tf,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float, float],
        float,
    ]:
        self.wait_ik_ready()
        candidates = self.request_graspgen_candidates()
        max_count = min(len(candidates), max(1, int(self.args.max_candidates)))
        ranked_candidates = self.rank_graspgen_candidates(candidates[:max_count], base_to_gripper_tf)
        print(
            f"GRASPGEN_CANDIDATE_TOTAL n={len(candidates)} tested={max_count} ik_filter={self.args.ik_filter} "
            f"ik_check_orientation={self.args.ik_check_orientation}",
            flush=True,
        )
        print(
            f"GRASPGEN_EE_ALIGNMENT graspgen_contact={fmt_xyz((self.args.graspgen_contact_x, self.args.graspgen_contact_y, self.args.graspgen_contact_z))} "
            f"target_contact={fmt_xyz((self.args.target_contact_x, self.args.target_contact_y, self.args.target_contact_z))} "
            f"adapter_xyz={fmt_xyz((self.args.graspgen_to_ee_x, self.args.graspgen_to_ee_y, self.args.graspgen_to_ee_z))} "
            f"adapter_rpy=({self.args.graspgen_to_ee_roll:.4f},{self.args.graspgen_to_ee_pitch:.4f},{self.args.graspgen_to_ee_yaw:.4f}) "
            f"auto_width_compensation={self.args.target_auto_width_compensation} "
            f"translation_auto={self.args._graspgen_to_ee_translation_auto}",
            flush=True,
        )
        if self.args.target_fixed_finger_contact_ee is not None and self.args.target_closing_axis_ee is not None:
            print(
                f"TARGET_WIDTH_COMP fixed_contact={fmt_xyz(self.args.target_fixed_finger_contact_ee)} "
                f"width_axis={fmt_xyz(self.args.target_closing_axis_ee)} "
                f"clearance={self.args.target_width_clearance_m:.4f} "
                f"width_limits=({self.args.target_width_min_m:.4f},{self.args.target_width_max_m:.4f}) "
                f"fallback={self.args.target_width_fallback_m:.4f} quality_min={self.args.target_width_quality_min:.3f}",
                flush=True,
            )
        elif self.args.target_auto_width_compensation and self.args._graspgen_to_ee_translation_auto:
            print(
                "TARGET_WIDTH_COMP enabled=False reason=missing_target_gripper_config "
                "hint=pass --robot-config with grasp_execution.target_gripper",
                flush=True,
            )

        for index, candidate, centroid_dist_camera, topdown_score in ranked_candidates:
            confidence = float(candidate.confidence)
            if confidence < self.args.min_grasp_confidence:
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} conf={confidence:.3f} "
                    f"reason=confidence_below_{self.args.min_grasp_confidence:.3f}",
                    flush=True,
                )
                continue
            if self.args.require_collision_free_grasp and not bool(candidate.collision_free):
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} conf={confidence:.3f} reason=not_collision_free",
                    flush=True,
                )
                continue

            try:
                (
                    t_base_ee,
                    t_base_graspgen,
                    t_camera_graspgen,
                    target_contact,
                    adapter_xyz,
                    width_reason,
                ) = self.graspgen_to_base_pose(candidate, base_to_gripper_tf)
            except ValueError as exc:
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} conf={confidence:.3f} reason={exc}",
                    flush=True,
                )
                continue
            approach, grasp, lift, quat, radius = self.graspgen_targets_from_pose(t_base_ee, t_base_graspgen)
            contact = self.graspgen_contact_point_base(t_base_graspgen)
            workspace_ok, workspace_reason = self._is_within_workspace(grasp, radius)
            height_ok, height_reason = self._graspgen_height_guard(approach, grasp, contact)
            camera_xyz = tuple(float(v) for v in t_camera_graspgen[:3, 3])
            camera_contact = self.graspgen_contact_point_camera(t_camera_graspgen)
            graspgen_axis_z = tuple(float(v) for v in t_base_graspgen[:3, 2])
            width, width_quality = candidate_target_width(candidate)
            centroid_dist_text = (
                f" centroid_dist_camera={centroid_dist_camera:.4f}" if centroid_dist_camera is not None else ""
            )
            print(
                f"GRASPGEN_CANDIDATE idx={index} conf={confidence:.3f} collision_free={bool(candidate.collision_free)} "
                f"graspgen_origin_camera={fmt_xyz(camera_xyz)} target_ee_grasp={fmt_xyz(grasp)} "
                f"contact_camera={fmt_xyz(camera_contact)} contact_base={fmt_xyz(contact)} "
                f"approach_axis_base={fmt_xyz(graspgen_axis_z)} target_ee_quat={fmt_quat(quat)} "
                f"topdown_score={topdown_score:.3f} "
                f"target_width={width:.4f} width_quality={width_quality:.3f} width_comp={width_reason} "
                f"target_contact={fmt_xyz(target_contact)} adapter_xyz={fmt_xyz(adapter_xyz)} "
                f"radius={radius:.4f}{centroid_dist_text}",
                flush=True,
            )
            if not workspace_ok:
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} "
                    f"radius={radius:.4f} reason={workspace_reason}",
                    flush=True,
                )
                continue
            if not height_ok:
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} "
                    f"contact={fmt_xyz(contact)} reason=height_guard_failed {height_reason}",
                    flush=True,
                )
                continue

            checks = [("approach", approach)]
            if self.args.require_grasp_ik:
                checks.append(("grasp", grasp))
            if self.args.require_lift_ik:
                checks.append(("lift", lift))

            failed = False
            for label, xyz in checks:
                ik_quat = quat if self.args.ik_check_orientation else None
                ik_ok, code = self.check_ik(f"graspgen_{index}_{label}", xyz, ik_quat)
                if not ik_ok:
                    print(
                        f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} "
                        f"reason=ik_failed_{label} code={code}",
                        flush=True,
                    )
                    failed = True
                    break
            if failed:
                continue

            print(
                f"GRASPGEN_CANDIDATE_ACCEPT idx={index} conf={confidence:.3f} "
                f"approach={fmt_xyz(approach)} grasp={fmt_xyz(grasp)} lift={fmt_xyz(lift)} quat={fmt_quat(quat)}",
                flush=True,
            )
            self.selected_target_contact_ee = tuple(float(v) for v in target_contact)
            self.selected_plan_contact_base = contact
            return approach, grasp, lift, quat, radius

        raise RuntimeError("No GraspGen candidate passed workspace and IK filters")

    def detection_to_base(self, detection: Detection2D, base_to_gripper_tf) -> tuple[float, float, float]:
        camera_point = (
            detection.centroid_xyz.x,
            detection.centroid_xyz.y,
            detection.centroid_xyz.z,
        )
        gripper_point = mat4_mul_point(self.handeye_matrix, camera_point)

        t = base_to_gripper_tf.transform.translation
        q = base_to_gripper_tf.transform.rotation
        rotated = quat_rotate((q.x, q.y, q.z, q.w), gripper_point)
        base_point = (rotated[0] + t.x, rotated[1] + t.y, rotated[2] + t.z)

        print(f"TARGET_GRIPPER x={gripper_point[0]:.4f} y={gripper_point[1]:.4f} z={gripper_point[2]:.4f}", flush=True)
        print(f"TARGET_BASE x={base_point[0]:.4f} y={base_point[1]:.4f} z={base_point[2]:.4f}", flush=True)
        return base_point

    def apply_target_offset(self, target_base: tuple[float, float, float]) -> tuple[float, float, float]:
        offset = (self.args.target_offset_x, self.args.target_offset_y, self.args.target_offset_z)
        if offset == (0.0, 0.0, 0.0):
            return target_base

        corrected = (
            target_base[0] + offset[0],
            target_base[1] + offset[1],
            target_base[2] + offset[2],
        )
        print(
            f"TARGET_OFFSET base_offset={fmt_xyz(offset)} corrected_base={fmt_xyz(corrected)}",
            flush=True,
        )
        return corrected

    def compute_gripper_targets(
        self, target_base: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
        bx, by, bz = target_base
        grasp = (
            bx - self.args.tcp_x,
            by - self.args.tcp_y,
            bz - self.args.tcp_z + self.args.tip_clearance_z,
        )
        approach = (grasp[0], grasp[1], grasp[2] + self.args.approach_lift)
        lift = (grasp[0], grasp[1], grasp[2] + self.args.final_lift)
        radius = math.sqrt(grasp[0] * grasp[0] + grasp[1] * grasp[1] + grasp[2] * grasp[2])

        print(
            f"GRIPPER_TARGETS approach={fmt_xyz(approach)} grasp={fmt_xyz(grasp)} "
            f"lift={fmt_xyz(lift)} radius={radius:.4f}",
            flush=True,
        )
        return approach, grasp, lift, radius

    def validate_pick_target(self, detection: Detection2D, grasp: tuple[float, float, float], radius: float) -> None:
        confidence = float(detection.confidence)
        if confidence < self.args.min_confidence_for_pick:
            raise RuntimeError(f"Detection confidence too low for pick: {confidence:.3f}")

        if self.args.allow_out_of_workspace:
            print("WORKSPACE_GUARD enabled=False", flush=True)
            return

        if radius > self.args.max_target_radius or abs(grasp[1]) > self.args.max_abs_y:
            raise RuntimeError(
                "Target outside guarded workspace: "
                f"radius={radius:.3f}, y={grasp[1]:.3f}, "
                f"limits radius<={self.args.max_target_radius:.3f}, abs_y<={self.args.max_abs_y:.3f}"
            )
        print("WORKSPACE_GUARD passed=True", flush=True)

    def _is_within_workspace(self, grasp: tuple[float, float, float], radius: float) -> tuple[bool, str]:
        if self.args.allow_out_of_workspace:
            return True, "workspace guard disabled"
        if radius > self.args.max_target_radius:
            return False, f"radius {radius:.3f} > {self.args.max_target_radius:.3f}"
        if abs(grasp[1]) > self.args.max_abs_y:
            return False, f"abs(y) {abs(grasp[1]):.3f} > {self.args.max_abs_y:.3f}"
        return True, "workspace ok"

    def _graspgen_height_guard(
        self,
        approach: tuple[float, float, float],
        grasp: tuple[float, float, float],
        contact: tuple[float, float, float],
    ) -> tuple[bool, str]:
        if self.args.allow_out_of_workspace:
            return True, "height guard disabled"
        failures = []
        if approach[2] < self.args.min_approach_z:
            failures.append(f"approach_z {approach[2]:.3f} < {self.args.min_approach_z:.3f}")
        if grasp[2] < self.args.min_grasp_z:
            failures.append(f"grasp_z {grasp[2]:.3f} < {self.args.min_grasp_z:.3f}")
        if contact[2] < self.args.min_contact_z:
            failures.append(f"contact_z {contact[2]:.3f} < {self.args.min_contact_z:.3f}")
        if failures:
            return False, "; ".join(failures)
        return True, "height ok"

    def _ik_timeout_duration(self):
        seconds = max(0.0, float(self.args.ik_timeout_s))
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1_000_000_000)
        return sec, nanosec

    def check_ik(
        self,
        label: str,
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float] | None = None,
    ) -> tuple[bool, int]:
        if not self.args.ik_filter or self.ik_client is None:
            return True, 1

        request = GetPositionIK.Request()
        request.ik_request.group_name = self.args.ik_group
        request.ik_request.ik_link_name = self.args.ee_frame
        request.ik_request.pose_stamped.header.frame_id = self.args.base_frame
        request.ik_request.pose_stamped.pose = make_pose(*xyz, quat_xyzw)
        request.ik_request.avoid_collisions = bool(self.args.ik_avoid_collisions)
        sec, nanosec = self._ik_timeout_duration()
        request.ik_request.timeout.sec = sec
        request.ik_request.timeout.nanosec = nanosec

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=max(1.0, self.args.ik_timeout_s + 1.0))
        if not future.done():
            print(
                f"IK_RESULT label={label} xyz={fmt_xyz(xyz)} quat={fmt_quat(quat_xyzw or (0.0, 0.0, 0.0, 1.0))} ok=False code=timeout",
                flush=True,
            )
            return False, -6

        response = future.result()
        code = int(response.error_code.val)
        ok = code == 1
        print(
            f"IK_RESULT label={label} xyz={fmt_xyz(xyz)} "
            f"quat={fmt_quat(quat_xyzw or (0.0, 0.0, 0.0, 1.0))} ok={ok} code={code}",
            flush=True,
        )
        return ok, code

    def _xy_offsets(self) -> list[tuple[float, float]]:
        radius = max(0.0, float(self.args.sample_xy_radius))
        step = max(0.0, float(self.args.sample_xy_step))
        if radius <= 0.0 or step <= 0.0:
            return [(0.0, 0.0)]

        count = max(1, int(math.floor(radius / step)))
        offsets: list[tuple[float, float]] = []
        for ix in range(-count, count + 1):
            for iy in range(-count, count + 1):
                dx = round(ix * step, 6)
                dy = round(iy * step, 6)
                if math.hypot(dx, dy) <= radius + 1e-9:
                    offsets.append((dx, dy))
        offsets.sort(key=lambda item: (math.hypot(item[0], item[1]), abs(item[0]) + abs(item[1]), item[0], item[1]))
        return offsets

    def generate_candidate_targets(
        self,
        primary_grasp: tuple[float, float, float],
    ) -> list[dict[str, object]]:
        z_offsets = parse_float_csv(self.args.sample_z_offsets)
        xy_offsets = self._xy_offsets()
        candidates: list[dict[str, object]] = []
        seen: set[tuple[float, float, float]] = set()

        for dz in z_offsets:
            for dx, dy in xy_offsets:
                grasp = (
                    primary_grasp[0] + dx,
                    primary_grasp[1] + dy,
                    primary_grasp[2] + dz,
                )
                key = (round(grasp[0], 5), round(grasp[1], 5), round(grasp[2], 5))
                if key in seen:
                    continue
                seen.add(key)

                approach = (grasp[0], grasp[1], grasp[2] + self.args.approach_lift)
                lift = (grasp[0], grasp[1], grasp[2] + self.args.final_lift)
                radius = math.sqrt(grasp[0] * grasp[0] + grasp[1] * grasp[1] + grasp[2] * grasp[2])
                candidates.append(
                    {
                        "offset": (dx, dy, dz),
                        "approach": approach,
                        "grasp": grasp,
                        "lift": lift,
                        "radius": radius,
                    }
                )
                if len(candidates) >= self.args.max_candidates:
                    return candidates
        return candidates

    def select_candidate(
        self,
        detection: Detection2D,
        primary_grasp: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
        confidence = float(detection.confidence)
        if confidence < self.args.min_confidence_for_pick:
            raise RuntimeError(f"Detection confidence too low for pick: {confidence:.3f}")

        self.wait_ik_ready()
        candidates = self.generate_candidate_targets(primary_grasp)
        print(f"CANDIDATE_TOTAL n={len(candidates)} ik_filter={self.args.ik_filter}", flush=True)

        for index, candidate in enumerate(candidates):
            approach = candidate["approach"]
            grasp = candidate["grasp"]
            lift = candidate["lift"]
            radius = float(candidate["radius"])
            offset = candidate["offset"]

            assert isinstance(approach, tuple)
            assert isinstance(grasp, tuple)
            assert isinstance(lift, tuple)
            assert isinstance(offset, tuple)

            workspace_ok, workspace_reason = self._is_within_workspace(grasp, radius)
            if not workspace_ok:
                print(
                    f"CANDIDATE_REJECT idx={index} offset={fmt_xyz(offset)} grasp={fmt_xyz(grasp)} "
                    f"radius={radius:.4f} reason={workspace_reason}",
                    flush=True,
                )
                continue

            checks = [("approach", approach)]
            if self.args.require_grasp_ik:
                checks.append(("grasp", grasp))
            if self.args.require_lift_ik:
                checks.append(("lift", lift))

            failed = False
            for label, xyz in checks:
                ik_ok, code = self.check_ik(f"candidate_{index}_{label}", xyz)
                if not ik_ok:
                    print(
                        f"CANDIDATE_REJECT idx={index} offset={fmt_xyz(offset)} grasp={fmt_xyz(grasp)} "
                        f"radius={radius:.4f} reason=ik_failed_{label} code={code}",
                        flush=True,
                    )
                    failed = True
                    break
            if failed:
                continue

            print(
                f"CANDIDATE_ACCEPT idx={index} offset={fmt_xyz(offset)} "
                f"approach={fmt_xyz(approach)} grasp={fmt_xyz(grasp)} lift={fmt_xyz(lift)} radius={radius:.4f}",
                flush=True,
            )
            self.selected_target_contact_ee = (self.args.tcp_x, self.args.tcp_y, self.args.tcp_z)
            self.selected_plan_contact_base = (
                grasp[0] + self.args.tcp_x,
                grasp[1] + self.args.tcp_y,
                grasp[2] + self.args.tcp_z,
            )
            return approach, grasp, lift, radius

        raise RuntimeError("No sampled candidate passed workspace and IK filters")

    def contact_base_from_tf(
        self,
        base_to_gripper_tf,
        contact_ee: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        t_base_gripper = transform_to_matrix(base_to_gripper_tf)
        contact = t_base_gripper @ np.array([contact_ee[0], contact_ee[1], contact_ee[2], 1.0], dtype=np.float64)
        return (float(contact[0]), float(contact[1]), float(contact[2]))

    def planned_contact_for_pose(
        self,
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float]:
        if quat_xyzw is None:
            return xyz
        contact_ee = self.selected_target_contact_ee or tuple(float(v) for v in _static_target_contact(self.args))
        contact = matrix_from_pose(xyz, quat_xyzw) @ np.array(
            [contact_ee[0], contact_ee[1], contact_ee[2], 1.0], dtype=np.float64
        )
        return (float(contact[0]), float(contact[1]), float(contact[2]))

    def correction_for_contact_alignment(
        self,
        commanded_xyz: tuple[float, float, float],
        planned_contact_base: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        base_to_gripper_tf = self.lookup_base_to_gripper()
        contact_ee = self.selected_target_contact_ee or tuple(float(v) for v in _static_target_contact(self.args))
        actual_contact = self.contact_base_from_tf(base_to_gripper_tf, contact_ee)
        error = sub_xyz(planned_contact_base, actual_contact)
        corrected = add_xyz(commanded_xyz, error)
        return corrected, error, norm_xyz(error)

    def realign_contact(
        self,
        phase: str,
        commanded_xyz: tuple[float, float, float],
        speed: float,
        quat_xyzw: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float]:
        if not self.args.contact_realign or quat_xyzw is None:
            return commanded_xyz

        current_command = commanded_xyz
        planned_contact_base = self.planned_contact_for_pose(commanded_xyz, quat_xyzw)
        self.current_plan_contact_base = planned_contact_base
        tolerance = max(0.0, float(self.args.contact_realign_tolerance))
        max_iterations = max(0, int(self.args.contact_realign_max_iterations))
        for iteration in range(max_iterations):
            corrected, correction, error_norm = self.correction_for_contact_alignment(
                current_command, planned_contact_base
            )
            print(
                f"CONTACT_REALIGN phase={phase} iter={iteration} error={fmt_xyz(correction)} "
                f"error_norm={error_norm:.4f} tolerance={tolerance:.4f} planned_contact={fmt_xyz(planned_contact_base)} "
                f"corrected={fmt_xyz(corrected)}",
                flush=True,
            )
            if error_norm <= tolerance:
                return current_command

            ok = self.run_task(
                f"banana_contact_realign_{phase}_{iteration}",
                f"contact realign {phase} {iteration}",
                [make_move_step(f"realign_{phase}_{iteration}", corrected, speed, quat_xyzw)],
            )
            if not ok:
                raise RuntimeError(f"Contact realignment failed during {phase}")
            current_command = corrected

        return current_command

    def sample_pick_diagnostics(
        self,
        label: str,
        commanded_xyz: tuple[float, float, float],
        commanded_quat_xyzw: tuple[float, float, float, float] | None = None,
        detect_target: bool = False,
    ) -> None:
        if not self.args.pick_diagnostics:
            return

        settle_s = max(0.0, float(self.args.pick_diagnostics_settle_s))
        if settle_s > 0.0:
            time.sleep(settle_s)

        base_to_gripper_tf = self.lookup_base_to_gripper()
        t = base_to_gripper_tf.transform.translation
        q = base_to_gripper_tf.transform.rotation
        actual_xyz = (float(t.x), float(t.y), float(t.z))
        actual_quat = (float(q.x), float(q.y), float(q.z), float(q.w))
        pose_delta = sub_xyz(actual_xyz, commanded_xyz)
        rot_delta_text = ""
        if commanded_quat_xyzw is not None:
            rot_delta_text = f" commanded_q={fmt_quat(commanded_quat_xyzw)} rot_delta_deg={quat_delta_deg(actual_quat, commanded_quat_xyzw):.2f}"
        print(
            f"PICK_DIAG_POSE label={label} commanded={fmt_xyz(commanded_xyz)} actual={fmt_xyz(actual_xyz)} "
            f"actual_minus_command={fmt_xyz(pose_delta)} norm={norm_xyz(pose_delta):.4f} "
            f"actual_q={fmt_quat(actual_quat)}{rot_delta_text}",
            flush=True,
        )

        contact_ee = self.selected_target_contact_ee or tuple(float(v) for v in _static_target_contact(self.args))
        actual_contact = self.contact_base_from_tf(base_to_gripper_tf, contact_ee)
        planned_contact = self.planned_contact_for_pose(commanded_xyz, commanded_quat_xyzw)
        self.current_plan_contact_base = planned_contact
        if planned_contact is not None:
            contact_delta = sub_xyz(actual_contact, planned_contact)
            print(
                f"PICK_DIAG_CONTACT label={label} contact_ee={fmt_xyz(contact_ee)} "
                f"planned={fmt_xyz(planned_contact)} actual={fmt_xyz(actual_contact)} "
                f"actual_minus_planned={fmt_xyz(contact_delta)} norm={norm_xyz(contact_delta):.4f}",
                flush=True,
            )
        else:
            print(
                f"PICK_DIAG_CONTACT label={label} contact_ee={fmt_xyz(contact_ee)} actual={fmt_xyz(actual_contact)}",
                flush=True,
            )

        if self.observed_target_base is not None:
            observed_target_minus_gripper = sub_xyz(self.observed_target_base, actual_xyz)
            observed_target_minus_contact = sub_xyz(self.observed_target_base, actual_contact)
            print(
                f"PICK_DIAG_OBS_TARGET label={label} observed_target_base={fmt_xyz(self.observed_target_base)} "
                f"observed_minus_gripper={fmt_xyz(observed_target_minus_gripper)} "
                f"gripper_norm={norm_xyz(observed_target_minus_gripper):.4f} "
                f"observed_minus_contact={fmt_xyz(observed_target_minus_contact)} "
                f"contact_norm={norm_xyz(observed_target_minus_contact):.4f}",
                flush=True,
            )

        if not detect_target or not self.args.pick_diagnostics_detect:
            return

        try:
            detection = self.detect_target()
            target_base = self.detection_to_base(detection, base_to_gripper_tf)
        except Exception as exc:
            print(f"PICK_DIAG_TARGET label={label} success=False error={exc}", flush=True)
            return

        target_minus_gripper = sub_xyz(target_base, actual_xyz)
        target_minus_contact = sub_xyz(target_base, actual_contact)
        self.observed_target_base = target_base
        print(
            f"PICK_DIAG_TARGET label={label} success=True target_base={fmt_xyz(target_base)} "
            f"target_minus_gripper={fmt_xyz(target_minus_gripper)} gripper_norm={norm_xyz(target_minus_gripper):.4f} "
            f"target_minus_contact={fmt_xyz(target_minus_contact)} contact_norm={norm_xyz(target_minus_contact):.4f}",
            flush=True,
        )

    def execute_pick(
        self,
        approach: tuple[float, float, float],
        grasp: tuple[float, float, float],
        lift: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float] | None = None,
    ) -> None:
        if self.args.detect_only:
            print("PICK skipped=True reason=detect_only", flush=True)
            return

        move_quat = quat_xyzw if self.args.execute_grasp_orientation else None
        if quat_xyzw is not None and move_quat is None:
            print(
                "EXECUTE_ORIENTATION enabled=False warning=width_compensation_assumes_graspgen_orientation "
                "effect=contact_alignment_may_be_invalid",
                flush=True,
            )

        if self.args.pick_diagnostics or self.args.contact_realign:
            task_id = "banana_graspgen_pick" if self.args.target_source == "graspgen" else "banana_tcp_compensated_pick"
            task_desc = (
                "pick target with GraspGen 6-DOF pose"
                if self.args.target_source == "graspgen"
                else "pick target with target-gripper TCP compensation"
            )

            ok = self.run_task(
                f"{task_id}_prepare",
                f"{task_desc}: prepare",
                [
                    make_gripper_step("open_gripper_before_pick", 1.0),
                    make_wait_step("settle_open_gripper", self.args.open_settle_s),
                ],
            )
            if not ok:
                raise RuntimeError("Pick task failed during prepare")

            ok = self.run_task(
                f"{task_id}_approach",
                f"{task_desc}: approach",
                [make_move_step("move_above_target", approach, self.args.approach_speed, move_quat)],
            )
            if not ok:
                raise RuntimeError("Pick task failed during approach")
            approach = self.realign_contact("approach", approach, self.args.approach_speed, move_quat)
            self.sample_pick_diagnostics(
                "approach",
                approach,
                commanded_quat_xyzw=move_quat,
                detect_target=False,
            )

            ok = self.run_task(
                f"{task_id}_grasp",
                f"{task_desc}: grasp",
                [make_move_step("descend_to_graspgen_pose", grasp, self.args.descend_speed, move_quat)],
            )
            if not ok:
                raise RuntimeError("Pick task failed during grasp")
            grasp = self.realign_contact("grasp", grasp, self.args.descend_speed, move_quat)
            lift = (grasp[0], grasp[1], grasp[2] + self.args.final_lift)
            self.sample_pick_diagnostics(
                "grasp",
                grasp,
                commanded_quat_xyzw=move_quat,
                detect_target=True,
            )

            ok = self.run_task(
                f"{task_id}_close",
                f"{task_desc}: close",
                [
                    make_gripper_step("close_gripper_on_target", 0.0),
                    make_wait_step("hold_target", self.args.hold_s),
                ],
            )
            if not ok:
                raise RuntimeError("Pick task failed during close")
            self.sample_pick_diagnostics(
                "close",
                grasp,
                commanded_quat_xyzw=move_quat,
                detect_target=True,
            )

            ok = self.run_task(
                f"{task_id}_lift",
                f"{task_desc}: lift",
                [make_move_step("lift_target", lift, self.args.lift_speed, move_quat)],
            )
            if not ok:
                raise RuntimeError("Pick task failed during lift")
            self.sample_pick_diagnostics(
                "lift",
                lift,
                commanded_quat_xyzw=move_quat,
                detect_target=False,
            )
            return

        steps = [
            make_gripper_step("open_gripper_before_pick", 1.0),
            make_wait_step("settle_open_gripper", self.args.open_settle_s),
            make_move_step("move_above_target", approach, self.args.approach_speed, move_quat),
            make_move_step("descend_to_graspgen_pose", grasp, self.args.descend_speed, move_quat),
            make_gripper_step("close_gripper_on_target", 0.0),
            make_wait_step("hold_target", self.args.hold_s),
            make_move_step("lift_target", lift, self.args.lift_speed, move_quat),
        ]
        task_id = "banana_graspgen_pick" if self.args.target_source == "graspgen" else "banana_tcp_compensated_pick"
        task_desc = (
            "pick target with GraspGen 6-DOF pose"
            if self.args.target_source == "graspgen"
            else "pick target with target-gripper TCP compensation"
        )
        ok = self.run_task(task_id, task_desc, steps)
        if not ok:
            raise RuntimeError("Pick task failed")


def main() -> None:
    args = parse_args()

    rclpy.init()
    node = BananaHandeyePickClient(args)
    try:
        print(
            f"CONFIG observe=({args.observe_x:.4f},{args.observe_y:.4f},{args.observe_z:.4f}) "
            f"prompt={args.prompt} target_source={args.target_source} handeye={args.handeye_json} "
            f"manipulation_service={args.manipulation_service} tcp=({args.tcp_x:.4f},{args.tcp_y:.4f},{args.tcp_z:.4f})",
            flush=True,
        )
        node.check_handeye_quality()
        node.wait_ready()
        node.move_to_observe()
        if args.observe_only:
            print("OBSERVE_ONLY success=True", flush=True)
            print("FLOW_RESULT success=True", flush=True)
            return
        base_to_gripper_tf = node.lookup_base_to_gripper()
        if args.target_source == "graspgen":
            approach, grasp, lift, quat, radius = node.select_graspgen_candidate(base_to_gripper_tf)
        else:
            detection = node.detect_target()
            target_base = node.detection_to_base(detection, base_to_gripper_tf)
            target_base = node.apply_target_offset(target_base)
            approach, grasp, lift, radius = node.compute_gripper_targets(target_base)
            approach, grasp, lift, radius = node.select_candidate(detection, grasp)
            quat = None
        node.execute_pick(approach, grasp, lift, quat)
        print("FLOW_RESULT success=True", flush=True)
    except Exception as exc:
        print(f"FLOW_RESULT success=False error={exc}", flush=True)
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
