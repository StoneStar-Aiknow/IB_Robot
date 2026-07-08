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
import struct
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
        "--execution-debug-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write SO101 execution-stage candidate JSON and a colored preview into the GraspGen debug output directory",
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
        "--centroid-source",
        choices=("volume", "surface"),
        default="volume",
        help="Which Detection2D centroid to use for ranking and target display: "
        "'volume' (convex-hull center, default) or 'surface' (visible-surface mean)",
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
        default=0.008,
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
        default=-0.080,
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
        default=0.008,
        help="Stop contact realignment when actual contact error is below this many meters",
    )
    parser.add_argument(
        "--contact-realign-max-iterations",
        type=int,
        default=4,
        help="Maximum correction moves at each safe realign phase",
    )
    parser.add_argument(
        "--pregrasp-realign-clearance",
        type=float,
        default=0.020,
        help="Place the target contact point this far above the observed object top for the final safe realign",
    )
    parser.add_argument(
        "--grasp-realign-max-xy-error",
        type=float,
        default=0.008,
        help="Warn after descent if low-height contact XY residual exceeds this; no XY realign is done at grasp",
    )
    parser.add_argument(
        "--grasp-residual-realign-xy-error",
        type=float,
        default=0.010,
        help="Retract to pregrasp and realign once if low-height contact XY residual exceeds this; <=0 disables it",
    )
    parser.add_argument(
        "--grasp-residual-abort-xy-error",
        type=float,
        default=0.030,
        help="Abort and retract only if low-height contact XY residual exceeds this hard safety limit; <=0 disables it",
    )
    parser.add_argument(
        "--max-execution-attempts",
        type=int,
        default=3,
        help="Maximum GraspGen execution candidates to try after retryable motion failures; <=0 tries all candidates",
    )
    parser.add_argument(
        "--retry-after-grasp-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Try another candidate after low-height contact residual abort; default stops after the safety retract",
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
    parser.add_argument(
        "--max-candidates",
        "--graspgen-max-candidates",
        dest="max_candidates",
        type=int,
        default=80,
        help="Maximum candidates to test; <=0 tests all returned candidates",
    )

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
        "--so101-tabletop-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject candidates whose SO101 gripper STL mesh intersects the fitted table plane",
    )
    parser.add_argument(
        "--so101-tabletop-clearance",
        type=float,
        default=0.0,
        help="Minimum signed distance in meters from SO101 gripper mesh to fitted table plane",
    )
    parser.add_argument(
        "--so101-tabletop-sweep-steps",
        type=int,
        default=5,
        help="Number of approach-to-grasp interpolation poses checked by the SO101 tabletop filter",
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
    parser.add_argument("--task-goal-timeout-s", type=float, default=30.0, help="Task action goal acceptance timeout")
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
    args.target_fixed_finger_margin_m = 0.0
    args.target_fixed_finger_margin_max_m = 0.0
    args.target_fixed_finger_margin_width_ref_m = 0.035
    args.target_fixed_finger_margin_width_gain = 0.0
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

    if "fixed_finger_margin_m" in target_gripper:
        args.target_fixed_finger_margin_m = float(target_gripper["fixed_finger_margin_m"])
        if args.target_fixed_finger_margin_m < 0.0:
            raise ValueError("grasp_execution.target_gripper.fixed_finger_margin_m must be non-negative")
    if "fixed_finger_margin_max_m" in target_gripper:
        args.target_fixed_finger_margin_max_m = float(target_gripper["fixed_finger_margin_max_m"])
        if args.target_fixed_finger_margin_max_m < 0.0:
            raise ValueError("grasp_execution.target_gripper.fixed_finger_margin_max_m must be non-negative")
    if "fixed_finger_margin_width_ref_m" in target_gripper:
        args.target_fixed_finger_margin_width_ref_m = float(target_gripper["fixed_finger_margin_width_ref_m"])
        if args.target_fixed_finger_margin_width_ref_m < 0.0:
            raise ValueError("grasp_execution.target_gripper.fixed_finger_margin_width_ref_m must be non-negative")
    if "fixed_finger_margin_width_gain" in target_gripper:
        args.target_fixed_finger_margin_width_gain = float(target_gripper["fixed_finger_margin_width_gain"])
        if args.target_fixed_finger_margin_width_gain < 0.0:
            raise ValueError("grasp_execution.target_gripper.fixed_finger_margin_width_gain must be non-negative")
    if (
        args.target_fixed_finger_margin_max_m > 0.0
        and args.target_fixed_finger_margin_max_m < args.target_fixed_finger_margin_m
    ):
        raise ValueError(
            "grasp_execution.target_gripper.fixed_finger_margin_max_m must be greater than or equal to "
            "fixed_finger_margin_m"
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
    base_fixed_finger_margin = float(args.target_fixed_finger_margin_m)
    max_fixed_finger_margin = float(args.target_fixed_finger_margin_max_m)
    if max_fixed_finger_margin <= 0.0:
        max_fixed_finger_margin = base_fixed_finger_margin
    width_ref = float(args.target_fixed_finger_margin_width_ref_m)
    width_gain = float(args.target_fixed_finger_margin_width_gain)
    requested_margin_extra = max(0.0, width_ref - width) * width_gain
    fixed_finger_margin = min(max(base_fixed_finger_margin + requested_margin_extra, 0.0), max_fixed_finger_margin)
    applied_margin_extra = max(0.0, fixed_finger_margin - base_fixed_finger_margin)
    center_offset = 0.5 * width_with_clearance + fixed_finger_margin
    contact = fixed_contact + width_axis * center_offset
    reason = (
        f"{source}:measured_width={measured_width:.4f}:used_width={width:.4f}:quality={quality:.3f}:"
        f"width_with_clearance={width_with_clearance:.4f}:"
        f"fixed_finger_margin_base={base_fixed_finger_margin:.4f}:"
        f"fixed_finger_margin_extra={applied_margin_extra:.4f}:"
        f"fixed_finger_margin={fixed_finger_margin:.4f}:"
        f"center_offset={center_offset:.4f}"
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


def _json_xyz(xyz: Iterable[float]) -> list[float]:
    return [round(float(v), 6) for v in xyz]


def _json_quat(quat_xyzw: Iterable[float] | None) -> list[float] | None:
    if quat_xyzw is None:
        return None
    return [round(float(v), 6) for v in quat_xyzw]


def add_xyz(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def sub_xyz(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def norm_xyz(xyz: tuple[float, float, float]) -> float:
    return math.sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2])


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class PickExecutionError(RuntimeError):
    def __init__(self, message: str, *, phase: str, retryable: bool):
        super().__init__(message)
        self.phase = phase
        self.retryable = retryable


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
        self.current_execution_candidate_index: int | None = None
        self.observed_target_base: tuple[float, float, float] | None = None
        self.observed_target_base_alt: tuple[float, float, float] | None = None
        self.current_plan_contact_base: tuple[float, float, float] | None = None
        self.realign_target_contact_base_by_phase: dict[str, tuple[float, float, float]] = {}
        self.last_graspgen_debug_output_dir: Path | None = None
        self.execution_debug_records: list[dict[str, object]] = []
        self.pick_diagnostic_records: list[dict[str, object]] = []
        self.last_t_base_camera: np.ndarray | None = None
        self._so101_table_plane: tuple[np.ndarray, float, float] | None = None
        self._so101_table_plane_checked = False

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
        goal_timeout_s = max(0.1, float(self.args.task_goal_timeout_s))
        rclpy.spin_until_future_complete(self, future, timeout_sec=goal_timeout_s)
        if not future.done():
            raise RuntimeError(f"Timed out sending task: {task_id} after {goal_timeout_s:.1f}s")
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
            f"surface_xyz=({detection.centroid_xyz.x:.4f},{detection.centroid_xyz.y:.4f},{detection.centroid_xyz.z:.4f}) "
            f"volume_xyz=({detection.volume_centroid_xyz.x:.4f},{detection.volume_centroid_xyz.y:.4f},{detection.volume_centroid_xyz.z:.4f}) "
            f"vol={detection.volume_m3 * 1e6:.1f}cm³",
            flush=True,
        )
        return detection

    def request_graspgen_response(
        self,
        debug_output_mode: str | None = None,
    ):
        request = PlanGrasp.Request()
        request.text_prompt = self.args.prompt
        request.confidence_threshold = self.args.confidence_threshold
        request.grasp_threshold = self.args.grasp_threshold
        request.debug_output_mode = debug_output_mode or self.args.debug_output_mode

        print(
            f"GRASPGEN_SEND prompt={self.args.prompt} detect_threshold={self.args.confidence_threshold} "
            f"grasp_threshold={self.args.grasp_threshold} debug_output_mode={request.debug_output_mode}",
            flush=True,
        )
        future = self.grasp_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.detect_timeout_s)
        if not future.done():
            raise RuntimeError("GraspGen planning timed out")

        response = future.result()
        self.print_graspgen_diagnostics(response)
        debug_output_dir = getattr(response, "debug_output_dir", "")
        self.last_graspgen_debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
        self._so101_table_plane = None
        self._so101_table_plane_checked = False
        return response

    def request_graspgen_candidates(self, base_to_gripper_tf=None):
        response = self.request_graspgen_response()
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

    @staticmethod
    def _contact_for_pose(
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
        contact_ee: Iterable[float],
    ) -> tuple[float, float, float]:
        contact = matrix_from_pose(xyz, quat_xyzw) @ np.array([*contact_ee, 1.0], dtype=np.float64)
        return (float(contact[0]), float(contact[1]), float(contact[2]))

    def _record_execution_candidate(
        self,
        *,
        index: int,
        confidence: float,
        stage: str,
        reason: str,
        collision_free: bool | None = None,
        topdown_score: float | None = None,
        centroid_dist_camera: float | None = None,
        approach: tuple[float, float, float] | None = None,
        grasp: tuple[float, float, float] | None = None,
        lift: tuple[float, float, float] | None = None,
        quat: tuple[float, float, float, float] | None = None,
        start: tuple[float, float, float] | None = None,
        start_quat: tuple[float, float, float, float] | None = None,
        target_contact: Iterable[float] | None = None,
        target_width_m: float | None = None,
        target_width_quality: float | None = None,
        grasp_mesh_min_z: float | None = None,
        so101_tabletop_clearance_m: float | None = None,
        adapter_xyz: Iterable[float] | None = None,
        width_reason: str = "",
        selected: bool = False,
    ) -> None:
        record: dict[str, object] = {
            "index": int(index),
            "confidence": round(float(confidence), 6),
            "stage": stage,
            "reason": reason,
            "selected": bool(selected),
        }
        if collision_free is not None:
            record["collision_free"] = bool(collision_free)
        if topdown_score is not None:
            record["topdown_score"] = round(float(topdown_score), 6)
        if centroid_dist_camera is not None:
            record["centroid_dist_camera"] = round(float(centroid_dist_camera), 6)
        if target_width_m is not None:
            record["target_width_m"] = round(float(target_width_m), 6)
        if target_width_quality is not None:
            record["target_width_quality"] = round(float(target_width_quality), 6)
        if grasp_mesh_min_z is not None:
            record["grasp_mesh_min_z"] = round(float(grasp_mesh_min_z), 6)
        if so101_tabletop_clearance_m is not None:
            record["so101_tabletop_clearance_m"] = round(float(so101_tabletop_clearance_m), 6)
        if adapter_xyz is not None:
            record["adapter_xyz"] = _json_xyz(adapter_xyz)
        if target_contact is not None:
            contact_tuple = tuple(float(v) for v in target_contact)
            record["target_contact_ee"] = _json_xyz(contact_tuple)
        else:
            contact_tuple = None
        if width_reason:
            record["width_reason"] = width_reason
        if approach is not None:
            record["approach"] = _json_xyz(approach)
        if grasp is not None:
            record["grasp"] = _json_xyz(grasp)
        if lift is not None:
            record["lift"] = _json_xyz(lift)
        if quat is not None:
            record["quat_xyzw"] = _json_quat(quat)
        if start is not None:
            record["start"] = _json_xyz(start)
        if start_quat is not None:
            record["start_quat_xyzw"] = _json_quat(start_quat)
        if contact_tuple is not None and start is not None and start_quat is not None:
            record["start_contact_base"] = _json_xyz(self._contact_for_pose(start, start_quat, contact_tuple))
        if contact_tuple is not None and quat is not None and approach is not None and grasp is not None:
            approach_contact = self._contact_for_pose(approach, quat, contact_tuple)
            grasp_contact = self._contact_for_pose(grasp, quat, contact_tuple)
            record["approach_contact_base"] = _json_xyz(approach_contact)
            record["grasp_contact_base"] = _json_xyz(grasp_contact)
            if lift is not None:
                record["lift_contact_base"] = _json_xyz(self._contact_for_pose(lift, quat, contact_tuple))
        self.execution_debug_records.append(record)

    def _write_execution_debug_outputs(self) -> None:
        if not self.args.execution_debug_preview or not self.execution_debug_records:
            return
        out_dir = self.last_graspgen_debug_output_dir
        if out_dir is None:
            return
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "frame_id": self.args.base_frame,
                "legend": {
                    "gray": "rejected before IK or by early execution-side guard",
                    "yellow": "passed workspace/height but rejected by IK",
                    "green": "IK passed candidate",
                    "red": "selected candidate",
                },
                "records": self.execution_debug_records,
            }
            (out_dir / "execution_candidates.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if self.pick_diagnostic_records:
                pick_payload = {
                    "frame_id": self.args.base_frame,
                    "gripper_frame": self.args.ee_frame,
                    "contact_frame": self.args.ee_frame,
                    "records": self.pick_diagnostic_records,
                }
                (out_dir / "pick_pose_diagnostics.json").write_text(
                    json.dumps(pick_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            self._render_execution_debug_preview_svg(out_dir / "grasp_preview_so101_execution.svg")
            self._render_execution_debug_preview_html(out_dir / "grasp_preview_so101_execution.html")
            self._render_execution_stage_overlay_svg(out_dir / "grasp_preview_execution_stages.svg")
            try:
                self._render_execution_debug_preview(out_dir / "grasp_preview_so101_execution.png")
            except Exception as exc:
                print(f"EXECUTION_DEBUG_PNG skipped=True error={exc}", flush=True)
            print(f"EXECUTION_DEBUG_OUTPUT dir={out_dir}", flush=True)
        except Exception as exc:
            print(f"EXECUTION_DEBUG_OUTPUT failed=True error={exc}", flush=True)

    def _execution_record_for_index(self, index: int) -> dict[str, object] | None:
        return next(
            (
                record
                for record in self.execution_debug_records
                if isinstance(record.get("index"), int) and int(record["index"]) == int(index)
            ),
            None,
        )

    def mark_execution_candidate_attempt(
        self,
        index: int,
        *,
        selected: bool,
        stage: str | None = None,
        reason: str | None = None,
    ) -> None:
        for record in self.execution_debug_records:
            record["selected"] = False
        record = self._execution_record_for_index(index)
        if record is None:
            return
        record["selected"] = bool(selected)
        if stage is not None:
            record["stage"] = stage
        if reason is not None:
            record["reason"] = reason

    def mark_execution_candidate_failed(self, index: int, phase: str, error: str) -> None:
        record = self._execution_record_for_index(index)
        if record is None:
            return
        record["selected"] = False
        record["stage"] = "execution_failed"
        record["reason"] = f"{phase}: {error}"

    @staticmethod
    def _read_ply_xyz(path: Path, max_points: int = 8000) -> np.ndarray:
        xyz, _ = BananaHandeyePickClient._read_ply_xyz_rgb(path, max_points=max_points)
        return xyz

    @staticmethod
    def _read_ply_xyz_rgb(path: Path, max_points: int = 8000) -> tuple[np.ndarray, np.ndarray]:
        if not path.exists():
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        with path.open("rb") as file:
            header_bytes = bytearray()
            while True:
                line = file.readline()
                if not line:
                    return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
                header_bytes.extend(line)
                if line.strip() == b"end_header":
                    break
            header = header_bytes.decode("ascii", errors="replace").splitlines()
            data_start = file.tell()
            data = file.read()

        vertex_count = 0
        fmt = ""
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header:
            parts = line.split()
            if not parts:
                continue
            if parts[:2] == ["format", "ascii"]:
                fmt = "ascii"
            elif parts[:2] == ["format", "binary_little_endian"]:
                fmt = "<"
            elif parts[:2] == ["format", "binary_big_endian"]:
                fmt = ">"
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[0] == "property" and len(parts) >= 3:
                properties.append((parts[1], parts[2]))
        if vertex_count <= 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        x_idx = next((i for i, (_, name) in enumerate(properties) if name == "x"), 0)
        y_idx = next((i for i, (_, name) in enumerate(properties) if name == "y"), 1)
        z_idx = next((i for i, (_, name) in enumerate(properties) if name == "z"), 2)
        r_idx = next((i for i, (_, name) in enumerate(properties) if name in ("red", "r")), None)
        g_idx = next((i for i, (_, name) in enumerate(properties) if name in ("green", "g")), None)
        b_idx = next((i for i, (_, name) in enumerate(properties) if name in ("blue", "b")), None)

        if fmt == "ascii":
            text = data.decode("ascii", errors="ignore").splitlines()
            rows = []
            colors = []
            step = max(1, vertex_count // max_points)
            for i, line in enumerate(text[:vertex_count]):
                if i % step != 0:
                    continue
                values = line.split()
                if len(values) >= 3:
                    rows.append([float(values[x_idx]), float(values[y_idx]), float(values[z_idx])])
                    if (
                        r_idx is not None
                        and g_idx is not None
                        and b_idx is not None
                        and len(values) > max(r_idx, g_idx, b_idx)
                    ):
                        colors.append([int(float(values[r_idx])), int(float(values[g_idx])), int(float(values[b_idx]))])
                    else:
                        colors.append([0, 0, 0])
            return np.asarray(rows, dtype=np.float32), np.asarray(colors, dtype=np.uint8)

        type_map = {
            "char": "b",
            "int8": "b",
            "uchar": "B",
            "uint8": "B",
            "short": "h",
            "int16": "h",
            "ushort": "H",
            "uint16": "H",
            "int": "i",
            "int32": "i",
            "uint": "I",
            "uint32": "I",
            "float": "f",
            "float32": "f",
            "double": "d",
            "float64": "d",
        }
        if fmt not in ("<", ">") or not properties:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        struct_fmt = fmt + "".join(type_map.get(prop_type, "f") for prop_type, _ in properties)
        row_size = struct.calcsize(struct_fmt)
        rows = []
        colors = []
        step = max(1, vertex_count // max_points)
        for i in range(0, vertex_count, step):
            start = i * row_size
            if start + row_size > len(data):
                break
            values = struct.unpack_from(struct_fmt, data, start)
            rows.append([float(values[x_idx]), float(values[y_idx]), float(values[z_idx])])
            if r_idx is not None and g_idx is not None and b_idx is not None:
                colors.append([int(values[r_idx]), int(values[g_idx]), int(values[b_idx])])
            else:
                colors.append([0, 0, 0])
        _ = data_start
        return np.asarray(rows, dtype=np.float32), np.asarray(colors, dtype=np.uint8)

    def _debug_object_cloud_layers(
        self, directory: Path, max_points: int = 16000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_path = directory / "object_cloud_raw.ply"
        completed_path = directory / "object_cloud_completed.ply"
        graspgen_input_path = directory / "object_cloud_graspgen_input.ply"
        if raw_path.exists() and completed_path.exists():
            raw_pts = self._read_ply_xyz(raw_path, max_points=max_points)
            completed_pts, completed_rgb = self._read_ply_xyz_rgb(completed_path, max_points=max_points)
            r = completed_rgb[:, 0]
            g = completed_rgb[:, 1]
            b = completed_rgb[:, 2]
            inpaint_mask = (r >= 240) & (g >= 140) & (g <= 200) & (b <= 50)
            prismatic_mask = (r <= 50) & (g >= 150) & (b >= 200)
            input_path = graspgen_input_path if graspgen_input_path.exists() else completed_path
            input_pts = self._read_ply_xyz(input_path, max_points=max_points)
            return raw_pts, completed_pts[inpaint_mask], completed_pts[prismatic_mask], input_pts
        fallback_pts = self._read_ply_xyz(directory / "object_cloud.ply", max_points=max_points)
        return (
            fallback_pts,
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            fallback_pts,
        )

    def object_top_z_base(self) -> tuple[float | None, str]:
        if self.last_graspgen_debug_output_dir is None:
            return None, "missing_debug_output_dir"
        if self.last_t_base_camera is None:
            return None, "missing_base_camera_transform"

        candidates = (
            ("object_cloud_raw", self.last_graspgen_debug_output_dir / "object_cloud_raw.ply"),
            ("object_cloud_completed", self.last_graspgen_debug_output_dir / "object_cloud_completed.ply"),
            ("object_cloud_graspgen_input", self.last_graspgen_debug_output_dir / "object_cloud_graspgen_input.ply"),
            ("object_cloud", self.last_graspgen_debug_output_dir / "object_cloud.ply"),
        )
        for label, path in candidates:
            if not path.exists():
                continue
            pts_camera = self._read_ply_xyz(path, max_points=30000)
            pts_base = self._transform_cloud(self.last_t_base_camera, pts_camera)
            if len(pts_base) == 0:
                continue
            top_z = float(np.percentile(pts_base[:, 2], 99.0))
            return top_z, f"{label}:p99"
        return None, "missing_object_cloud"

    @staticmethod
    def _preview_lines_for_pose(pose_4x4: np.ndarray, gripper_name: str) -> list[np.ndarray]:
        try:
            from grasp_gen.robot import load_control_points_for_visualization

            lines = []
            for ctrl_pts in load_control_points_for_visualization(gripper_name):
                pts = np.asarray(ctrl_pts, dtype=np.float32)
                pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
                lines.append((pose_4x4[:3, :] @ pts_h.T).T)
            return lines
        except Exception:
            origin = pose_4x4[:3, 3]
            scale = 0.055
            return [
                np.vstack([origin, origin + pose_4x4[:3, 0] * scale]),
                np.vstack([origin, origin + pose_4x4[:3, 1] * scale]),
                np.vstack([origin, origin + pose_4x4[:3, 2] * scale]),
            ]

    def _render_execution_stage_overlay_svg(self, path: Path) -> None:
        result_path = path.parent / "grasp_result.json"
        if not result_path.exists():
            return
        result = json.loads(result_path.read_text(encoding="utf-8"))
        grasps = result.get("grasps", [])
        if not isinstance(grasps, list) or not grasps:
            return

        records_by_index = {
            int(record["index"]): record
            for record in self.execution_debug_records
            if isinstance(record.get("index"), int)
        }
        stage_color = {
            "selected": "#dc2626",
            "height_rejected": "#2563eb",
            "workspace_rejected": "#f97316",
            "ik_rejected": "#f59e0b",
            "execution_failed": "#ec4899",
            "adapter_rejected": "#9333ea",
            "confidence_rejected": "#64748b",
            "collision_rejected": "#92400e",
            "not_tested": "#94a3b8",
        }
        stage_label = {
            "selected": "selected/executed",
            "height_rejected": "height guard rejected",
            "workspace_rejected": "workspace rejected",
            "ik_rejected": "IK rejected",
            "execution_failed": "execution failed",
            "adapter_rejected": "adapter rejected",
            "confidence_rejected": "confidence rejected",
            "collision_rejected": "collision flag rejected",
            "not_tested": "not tested after first-pass selection",
        }
        gripper_name = str(result.get("visualization_gripper") or result.get("gripper") or "robotiq_2f_140")
        object_pts, inpaint_pts, prismatic_pts, _ = self._debug_object_cloud_layers(path.parent)

        line_sets: list[tuple[int, str, float, bool, list[np.ndarray], np.ndarray]] = []
        bounds_parts = []
        if len(object_pts):
            bounds_parts.append(object_pts[:, :2])
        if len(inpaint_pts):
            bounds_parts.append(inpaint_pts[:, :2])
        if len(prismatic_pts):
            bounds_parts.append(prismatic_pts[:, :2])
        for fallback_idx, grasp in enumerate(grasps):
            if not isinstance(grasp, dict):
                continue
            idx = int(grasp.get("index", fallback_idx))
            pose_values = grasp.get("pose_4x4_rowmajor")
            if not isinstance(pose_values, list) or len(pose_values) != 16:
                continue
            pose = np.asarray(pose_values, dtype=np.float32).reshape(4, 4)
            record = records_by_index.get(idx)
            selected = bool(record.get("selected")) if record is not None else False
            stage = (
                "selected"
                if selected
                else str(record.get("stage", "not_tested") if record is not None else "not_tested")
            )
            confidence = float(grasp.get("confidence", 0.0))
            lines = self._preview_lines_for_pose(pose, gripper_name)
            for line in lines:
                if len(line):
                    bounds_parts.append(np.asarray(line[:, :2], dtype=np.float32))
            bounds_parts.append(pose[:2, 3].reshape(1, 2))
            line_sets.append((idx, stage, confidence, selected, lines, pose[:3, 3].copy()))
        if not line_sets:
            return

        bounds = np.vstack(bounds_parts) if bounds_parts else np.zeros((1, 2), dtype=np.float32)
        mins = bounds.min(axis=0)
        maxs = bounds.max(axis=0)
        span = np.maximum(maxs - mins, 0.03)
        mins -= span * 0.08
        maxs += span * 0.08
        width = 1100
        height = 860
        plot_x = 55
        plot_y = 105
        plot_w = 760
        plot_h = 700

        def project_xy(point: Iterable[float]) -> tuple[float, float]:
            values = list(point)
            x = float(values[0])
            y = float(values[1])
            px = plot_x + (x - mins[0]) / max(float(maxs[0] - mins[0]), 1e-6) * plot_w
            py = plot_y + (y - mins[1]) / max(float(maxs[1] - mins[1]), 1e-6) * plot_h
            return px, py

        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="55" y="38" font-size="22" font-family="monospace">Grasp preview with execution-stage colors</text>',
            '<text x="55" y="65" font-size="13" font-family="monospace">Same camera X-Y projection as service preview. Colors show execution-side filter result for server-returned poses.</text>',
            f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
        ]
        if len(object_pts):
            sample = object_pts
            for point in sample:
                x, y = project_xy(point)
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="0.75" fill="#22c55e" opacity="0.42"/>')
        if len(inpaint_pts):
            for point in inpaint_pts:
                x, y = project_xy(point)
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.05" fill="#f59e0b" opacity="0.72"/>')
        if len(prismatic_pts):
            for point in prismatic_pts:
                x, y = project_xy(point)
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="0.9" fill="#06b6d4" opacity="0.5"/>')

        for idx, stage, confidence, selected, lines, position in line_sets:
            color = stage_color.get(stage, stage_color["not_tested"])
            stroke_width = 4.2 if selected else 2.0
            opacity = 1.0 if selected else 0.78
            for line in lines:
                if len(line) < 2:
                    continue
                points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (project_xy(point) for point in line))
                elements.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity:.2f}"/>'
                )
            x, y = project_xy(position)
            if selected:
                star = " ".join(
                    f"{x + math.cos(angle) * (13 if i % 2 == 0 else 6):.1f},{y + math.sin(angle) * (13 if i % 2 == 0 else 6):.1f}"
                    for i, angle in enumerate(np.linspace(-math.pi / 2, 3 * math.pi / 2, 10, endpoint=False))
                )
                elements.append(f'<polygon points="{star}" fill="{color}" stroke="white" stroke-width="1.5"/>')
                label = f"EXEC #{idx} {stage_label.get(stage, stage)} conf={confidence:.3f}"
                elements.append(
                    f'<text x="{x + 14:.1f}" y="{y - 12:.1f}" font-size="12" font-family="monospace" fill="#991b1b" font-weight="bold">{label}</text>'
                )
            else:
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="white"/>')
                elements.append(
                    f'<text x="{x + 6:.1f}" y="{y - 6:.1f}" font-size="10" font-family="monospace" fill="#0f172a">#{idx}</text>'
                )

        legend_x = 845
        legend_y = 115
        elements.append(
            f'<text x="{legend_x}" y="{legend_y - 22}" font-size="16" font-family="monospace">Legend</text>'
        )
        for i, (stage, label) in enumerate(stage_label.items()):
            y = legend_y + i * 28
            elements.append(f'<rect x="{legend_x}" y="{y - 12}" width="18" height="18" fill="{stage_color[stage]}"/>')
            elements.append(
                f'<text x="{legend_x + 28}" y="{y + 2}" font-size="12" font-family="monospace" fill="#0f172a">{label}</text>'
            )
        elements.append(
            f'<text x="{legend_x}" y="{legend_y + 250}" font-size="12" font-family="monospace" fill="#475569">Green points: raw object cloud</text>'
        )
        elements.append(
            f'<text x="{legend_x}" y="{legend_y + 272}" font-size="12" font-family="monospace" fill="#475569">Orange points: mask-depth inpaint</text>'
        )
        elements.append(
            f'<text x="{legend_x}" y="{legend_y + 294}" font-size="12" font-family="monospace" fill="#475569">Cyan points: prismatic side extrude</text>'
        )
        elements.append(
            f'<text x="{legend_x}" y="{legend_y + 316}" font-size="12" font-family="monospace" fill="#475569">Purple points: completed table surface</text>'
        )
        elements.append(
            f'<text x="{legend_x}" y="{legend_y + 338}" font-size="12" font-family="monospace" fill="#475569">Star: final executed pose</text>'
        )
        elements.append(
            f'<text x="{legend_x}" y="{legend_y + 360}" font-size="12" font-family="monospace" fill="#475569">Gray not_tested = after first-pass stop</text>'
        )
        elements.append(
            f'<text x="{plot_x + plot_w - 45}" y="{plot_y + plot_h + 28}" font-size="12" font-family="monospace">X (m)</text>'
        )
        elements.append(
            f'<text x="{plot_x + 8}" y="{plot_y + 18}" font-size="12" font-family="monospace">Y (m, camera-down)</text>'
        )
        elements.append("</svg>")
        path.write_text("\n".join(elements), encoding="utf-8")

    @staticmethod
    def _transform_points(xyz: Iterable[float], quat_xyzw: Iterable[float], points_ee: np.ndarray) -> np.ndarray:
        transform = matrix_from_pose(tuple(float(v) for v in xyz), tuple(float(v) for v in quat_xyzw))
        points_h = np.hstack([points_ee.astype(np.float64), np.ones((len(points_ee), 1), dtype=np.float64)])
        return (transform[:3, :] @ points_h.T).T

    @staticmethod
    def _transform_cloud(transform: np.ndarray | None, points: np.ndarray) -> np.ndarray:
        if transform is None or len(points) == 0:
            return points
        points_h = np.hstack([points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)])
        return (transform[:3, :] @ points_h.T).T.astype(np.float32)

    @staticmethod
    def _matrix_from_xyz_rpy(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = R.from_euler("xyz", [float(v) for v in rpy]).as_matrix()
        matrix[:3, 3] = [float(v) for v in xyz]
        return matrix

    @staticmethod
    def _read_stl_triangles(path: Path, max_triangles: int = 2500) -> tuple[np.ndarray, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        triangles: np.ndarray
        if len(data) >= 84:
            tri_count = struct.unpack_from("<I", data, 80)[0]
            expected_size = 84 + tri_count * 50
            if tri_count > 0 and expected_size == len(data):
                stl_dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
                raw = np.frombuffer(data, dtype=stl_dtype, count=tri_count, offset=84)
                triangles = raw["vertices"].astype(np.float64)
            else:
                triangles = np.zeros((0, 3, 3), dtype=np.float64)
        else:
            triangles = np.zeros((0, 3, 3), dtype=np.float64)
        if len(triangles) == 0:
            vertices = []
            for line in data.decode("ascii", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) == 4 and parts[0] == "vertex":
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(vertices) >= 3:
                triangles = np.asarray(vertices[: len(vertices) // 3 * 3], dtype=np.float64).reshape(-1, 3, 3)
        if len(triangles) == 0:
            raise ValueError(f"No STL triangles found in {path}")
        step = max(1, int(math.ceil(len(triangles) / max_triangles)))
        triangles = triangles[::step]
        vertices = triangles.reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        return vertices, faces

    @classmethod
    def _so101_width_to_jaw_angle(cls, width_m: float | None) -> float:
        if width_m is None:
            return 0.45
        normalized = (float(width_m) - 0.008) / (0.080 - 0.008)
        return max(0.0, min(1.0, normalized))

    @classmethod
    def _transform_mesh(cls, transform: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        vertices_h = np.hstack([vertices.astype(np.float64), np.ones((len(vertices), 1), dtype=np.float64)])
        return (transform[:3, :] @ vertices_h.T).T

    @classmethod
    def _so101_gripper_meshes(
        cls,
        xyz: Iterable[float],
        quat_xyzw: Iterable[float],
        width_m: float | None,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        mesh_dir = Path(__file__).resolve().parents[1] / "src/robot_description/meshes/lerobot/so101"
        t_base_gripper = matrix_from_pose(tuple(float(v) for v in xyz), tuple(float(v) for v in quat_xyzw))
        jaw_angle = cls._so101_width_to_jaw_angle(width_m)

        t_gripper_visual = cls._matrix_from_xyz_rpy(
            (5.55112e-17, -0.000218214, 0.000949706),
            (-3.14159, -5.55112e-17, -9.17912e-24),
        )
        t_gripper_jaw = cls._matrix_from_xyz_rpy((0.0202, 0.0188, -0.0234), (1.5708, 0.209440, 0.000001))
        t_jaw_motion = cls._matrix_from_xyz_rpy((0.0, 0.0, 0.0), (0.0, 0.0, jaw_angle))
        t_jaw_visual = cls._matrix_from_xyz_rpy((-5.55112e-17, -1.94746e-17, 0.0189), (9.53145e-17, -4.66093e-24, 0.0))

        parts = [
            (
                "fixed gripper mesh",
                mesh_dir / "wrist_roll_follower_so101_v1.stl",
                t_base_gripper @ t_gripper_visual,
            ),
            (
                "moving jaw mesh",
                mesh_dir / "moving_jaw_so101_v1.stl",
                t_base_gripper @ t_gripper_jaw @ t_jaw_motion @ t_jaw_visual,
            ),
        ]
        meshes = []
        for name, mesh_path, transform in parts:
            vertices, faces = cls._read_stl_triangles(mesh_path)
            meshes.append((name, cls._transform_mesh(transform, vertices), faces))
        return meshes

    @classmethod
    def _so101_gripper_mesh_min_z(
        cls,
        xyz: Iterable[float],
        quat_xyzw: Iterable[float],
        width_m: float | None,
    ) -> float | None:
        try:
            meshes = cls._so101_gripper_meshes(xyz, quat_xyzw, width_m)
        except Exception as exc:
            print(f"SO101_MESH_HEIGHT_CHECK skipped=True error={exc}", flush=True)
            return None
        min_z = min(float(vertices[:, 2].min()) for _, vertices, _ in meshes if len(vertices))
        return min_z

    def _load_so101_table_plane(self) -> tuple[np.ndarray, float, float] | None:
        if self._so101_table_plane_checked:
            return self._so101_table_plane
        self._so101_table_plane_checked = True
        if self.last_graspgen_debug_output_dir is None:
            print("SO101_TABLETOP_FILTER skipped=True reason=no_debug_output_dir", flush=True)
            return None
        scene_path = self.last_graspgen_debug_output_dir / "scene_cloud.ply"
        object_path = self.last_graspgen_debug_output_dir / "object_cloud.ply"
        if not scene_path.exists():
            print(f"SO101_TABLETOP_FILTER skipped=True reason=missing_scene_cloud path={scene_path}", flush=True)
            return None
        scene_pts = self._transform_cloud(self.last_t_base_camera, self._read_ply_xyz(scene_path, max_points=30000))
        object_pts = self._transform_cloud(self.last_t_base_camera, self._read_ply_xyz(object_path, max_points=16000))
        if len(scene_pts) < 100:
            print(f"SO101_TABLETOP_FILTER skipped=True reason=too_few_scene_points n={len(scene_pts)}", flush=True)
            return None
        positive_reference = object_pts.mean(axis=0) if len(object_pts) else None
        try:
            from manipulation_service.graspgen_wrapper import fit_table_plane_ransac

            fit = fit_table_plane_ransac(
                scene_pts,
                positive_reference=positive_reference,
                distance_threshold=0.006,
                min_inlier_ratio=0.15,
            )
        except Exception as exc:
            print(f"SO101_TABLETOP_FILTER skipped=True reason=plane_fit_failed error={exc}", flush=True)
            return None
        if fit.plane is None:
            print(
                "SO101_TABLETOP_FILTER skipped=True reason=no_table_plane "
                f"best_inlier_ratio={fit.best_inlier_ratio:.3f} failure={fit.failure_reason}",
                flush=True,
            )
            return None
        self._so101_table_plane = (
            np.asarray(fit.plane.normal, dtype=np.float64),
            float(fit.plane.d),
            float(fit.plane.inlier_ratio),
        )
        normal, d, inlier_ratio = self._so101_table_plane
        print(
            "SO101_TABLETOP_FILTER plane_found=True "
            f"normal={fmt_xyz(normal)} d={d:.4f} inlier_ratio={inlier_ratio:.3f}",
            flush=True,
        )
        return self._so101_table_plane

    def _so101_tabletop_clearance(
        self,
        approach: tuple[float, float, float],
        grasp: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
        width_m: float | None,
    ) -> float | None:
        if not self.args.so101_tabletop_filter:
            return None
        plane = self._load_so101_table_plane()
        if plane is None:
            return None
        normal, d, _ = plane
        steps = max(1, int(self.args.so101_tabletop_sweep_steps))
        min_clearance = math.inf
        for alpha in np.linspace(0.0, 1.0, steps + 1):
            xyz = tuple(float((1.0 - alpha) * approach[i] + alpha * grasp[i]) for i in range(3))
            try:
                meshes = self._so101_gripper_meshes(xyz, quat_xyzw, width_m)
            except Exception as exc:
                print(f"SO101_TABLETOP_FILTER skipped=True reason=mesh_failed error={exc}", flush=True)
                return None
            for _, vertices, _ in meshes:
                if len(vertices):
                    signed = vertices @ normal + d
                    min_clearance = min(min_clearance, float(np.min(signed)))
        return min_clearance if math.isfinite(min_clearance) else None

    @classmethod
    def _so101_gripper_wireframe_lines(
        cls,
        xyz: Iterable[float],
        quat_xyzw: Iterable[float],
        contact_ee: Iterable[float],
        width_m: float | None,
    ) -> list[np.ndarray]:
        contact = np.asarray([float(v) for v in contact_ee], dtype=np.float64)
        width = float(width_m) if width_m is not None else 0.035
        half_width = max(width * 0.5, 0.008)
        finger_half_thick_y = 0.006
        finger_half_thick_x = 0.003
        finger_base_z = contact[2] + 0.055
        finger_tip_z = contact[2] - 0.010

        lines_ee = [
            np.array([[0.0, 0.0, 0.0], contact], dtype=np.float64),
            np.array([[-half_width, 0.0, finger_base_z], [half_width, 0.0, finger_base_z]], dtype=np.float64),
            np.array(
                [[contact[0] - half_width, 0.0, contact[2]], [contact[0] + half_width, 0.0, contact[2]]],
                dtype=np.float64,
            ),
        ]
        for side in (-1.0, 1.0):
            cx = contact[0] + side * half_width
            x0 = cx - finger_half_thick_x
            x1 = cx + finger_half_thick_x
            y0 = -finger_half_thick_y
            y1 = finger_half_thick_y
            z0 = finger_base_z
            z1 = finger_tip_z
            corners = np.array(
                [
                    [x0, y0, z0],
                    [x1, y0, z0],
                    [x1, y1, z0],
                    [x0, y1, z0],
                    [x0, y0, z1],
                    [x1, y0, z1],
                    [x1, y1, z1],
                    [x0, y1, z1],
                ],
                dtype=np.float64,
            )
            for edge in (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ):
                lines_ee.append(corners[list(edge)])
        return [cls._transform_points(xyz, quat_xyzw, line) for line in lines_ee]

    @staticmethod
    def _plotly_line_trace(line: np.ndarray, *, name: str, color: str, width: int = 5, showlegend: bool = False):
        import plotly.graph_objects as go

        return go.Scatter3d(
            x=line[:, 0],
            y=line[:, 1],
            z=line[:, 2],
            mode="lines",
            name=name,
            line={"color": color, "width": width},
            showlegend=showlegend,
            hoverinfo="skip",
        )

    def _render_execution_debug_preview_html(self, path: Path) -> None:
        try:
            import plotly.graph_objects as go
        except Exception as exc:
            print(f"EXECUTION_DEBUG_HTML skipped=True error={exc}", flush=True)
            return

        if not self.execution_debug_records:
            return
        object_pts_camera, inpaint_pts_camera, prismatic_pts_camera, graspgen_input_camera = (
            self._debug_object_cloud_layers(path.parent, max_points=16000)
        )
        object_pts = self._transform_cloud(self.last_t_base_camera, object_pts_camera)
        inpaint_pts = self._transform_cloud(self.last_t_base_camera, inpaint_pts_camera)
        prismatic_pts = self._transform_cloud(self.last_t_base_camera, prismatic_pts_camera)
        graspgen_input_pts = self._transform_cloud(self.last_t_base_camera, graspgen_input_camera)
        scene_raw_xyz, scene_raw_rgb = self._read_ply_xyz_rgb(path.parent / "scene_cloud.ply", max_points=300000)
        table_completion_camera = np.zeros((0, 3), dtype=np.float32)
        if len(scene_raw_xyz) > 0 and scene_raw_rgb is not None and len(scene_raw_rgb) == len(scene_raw_xyz):
            table_mask = (
                (scene_raw_rgb[:, 0] >= 140)
                & (scene_raw_rgb[:, 0] <= 200)
                & (scene_raw_rgb[:, 1] >= 60)
                & (scene_raw_rgb[:, 1] <= 120)
                & (scene_raw_rgb[:, 2] >= 220)
            )
            if table_mask.sum() > 0 and table_mask.sum() < len(scene_raw_xyz):
                gray_xyz = scene_raw_xyz[~table_mask]
                table_completion_camera = scene_raw_xyz[table_mask]
                gray_target = max(0, 50000 - int(table_mask.sum()))
                if len(gray_xyz) > gray_target:
                    idx = np.linspace(0, len(gray_xyz) - 1, gray_target, dtype=int)
                    gray_xyz = gray_xyz[idx]
                scene_pts_camera = gray_xyz
            else:
                scene_pts_camera = scene_raw_xyz
        else:
            scene_pts_camera = scene_raw_xyz
        scene_pts = self._transform_cloud(self.last_t_base_camera, scene_pts_camera)
        table_completion_pts = self._transform_cloud(self.last_t_base_camera, table_completion_camera)
        selected_record = next((record for record in self.execution_debug_records if record.get("selected")), None)
        selected_gripper_meshes: list[tuple[str, str, list[tuple[str, np.ndarray, np.ndarray]]]] = []
        selected_gripper_lines: list[tuple[str, str, list[np.ndarray]]] = []
        pick_records = [record for record in self.pick_diagnostic_records if isinstance(record, dict)]

        def record_xyz(record: dict[str, object], key: str) -> list[float] | None:
            value = record.get(key)
            if isinstance(value, list) and len(value) == 3:
                return [float(v) for v in value]
            return None

        def target_detection_xyz(record: dict[str, object]) -> list[float] | None:
            detection = record.get("target_detection")
            if isinstance(detection, dict):
                value = detection.get("target_base")
                if isinstance(value, list) and len(value) == 3:
                    return [float(v) for v in value]
            return None

        roi_parts = []
        if len(object_pts):
            if len(object_pts) > 20:
                roi_parts.append(
                    np.vstack(
                        [
                            np.percentile(object_pts, 2.0, axis=0),
                            np.percentile(object_pts, 98.0, axis=0),
                        ]
                    )
                )
            else:
                roi_parts.append(object_pts)
        if len(inpaint_pts):
            roi_parts.append(inpaint_pts)
        if len(prismatic_pts):
            roi_parts.append(prismatic_pts)
        if len(graspgen_input_pts):
            roi_parts.append(graspgen_input_pts)
        if len(table_completion_pts):
            roi_parts.append(table_completion_pts)
        for record in self.execution_debug_records:
            for key in (
                "start_contact_base",
                "approach_contact_base",
                "pregrasp_realign_contact_base",
                "grasp_contact_base",
                "lift_contact_base",
            ):
                value = record.get(key)
                if isinstance(value, list) and len(value) == 3:
                    roi_parts.append(np.asarray([[float(v) for v in value]], dtype=np.float64))
        for record in pick_records:
            for key in (
                "commanded",
                "actual",
                "planned_contact_base",
                "commanded_pose_contact_base",
                "actual_contact_base",
            ):
                value = record_xyz(record, key)
                if value is not None:
                    roi_parts.append(np.asarray([value], dtype=np.float64))
        if selected_record is not None:
            target_contact = selected_record.get("target_contact_ee")
            width = selected_record.get("target_width_m")
            if isinstance(target_contact, list):
                for label, key, quat_key, color in (
                    ("SO101 start gripper (current)", "start", "start_quat_xyzw", "#64748b"),
                    ("SO101 approach gripper", "approach", "quat_xyzw", "#0ea5e9"),
                    ("SO101 pregrasp realign gripper", "pregrasp_realign", "quat_xyzw", "#10b981"),
                    ("SO101 grasp gripper", "grasp", "quat_xyzw", "#ef4444"),
                ):
                    pose = selected_record.get(key)
                    quat = selected_record.get(quat_key)
                    if isinstance(pose, list) and len(pose) == 3 and isinstance(quat, list) and len(quat) == 4:
                        width_value = float(width) if isinstance(width, float | int) else None
                        try:
                            meshes = self._so101_gripper_meshes(pose, quat, width_value)
                            selected_gripper_meshes.append((label, color, meshes))
                            roi_parts.extend(vertices for _, vertices, _ in meshes)
                        except Exception as exc:
                            print(f"EXECUTION_DEBUG_HTML mesh_fallback=True label={label} error={exc}", flush=True)
                            lines = self._so101_gripper_wireframe_lines(pose, quat, target_contact, width_value)
                            selected_gripper_lines.append((label, color, lines))
                            roi_parts.extend(lines)
        if roi_parts:
            roi = np.vstack(roi_parts).astype(np.float64)
            roi_min = roi.min(axis=0)
            roi_max = roi.max(axis=0)
            roi_span = np.maximum(roi_max - roi_min, np.array([0.08, 0.08, 0.06], dtype=np.float64))
            roi_min -= roi_span * 0.45
            roi_max += roi_span * 0.45

            def crop(points: np.ndarray) -> np.ndarray:
                if len(points) == 0:
                    return points
                mask = np.all((points >= roi_min) & (points <= roi_max), axis=1)
                return points[mask]

            object_pts = crop(object_pts)
            inpaint_pts = crop(inpaint_pts)
            prismatic_pts = crop(prismatic_pts)
            graspgen_input_pts = crop(graspgen_input_pts)
            table_completion_pts = crop(table_completion_pts)
            scene_pts = crop(scene_pts)
        else:
            roi_min = np.array([-0.1, -0.1, -0.05], dtype=np.float64)
            roi_max = np.array([0.1, 0.1, 0.15], dtype=np.float64)
        traces = []
        if len(scene_pts):
            traces.append(
                go.Scatter3d(
                    x=scene_pts[:, 0],
                    y=scene_pts[:, 1],
                    z=scene_pts[:, 2],
                    mode="markers",
                    name=f"scene cloud ({len(scene_pts)})",
                    marker={"size": 1.1, "color": "rgba(150,150,150,0.18)"},
                    hoverinfo="skip",
                )
            )
        if len(table_completion_pts):
            traces.append(
                go.Scatter3d(
                    x=table_completion_pts[:, 0],
                    y=table_completion_pts[:, 1],
                    z=table_completion_pts[:, 2],
                    mode="markers",
                    name=f"completed table surface ({len(table_completion_pts)})",
                    marker={"size": 1.8, "color": "rgba(168,85,247,0.72)"},
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
                    name=f"raw object cloud ({len(object_pts)})",
                    marker={"size": 1.8, "color": "rgba(34,197,94,0.75)"},
                    hoverinfo="skip",
                )
            )
        if len(inpaint_pts):
            traces.append(
                go.Scatter3d(
                    x=inpaint_pts[:, 0],
                    y=inpaint_pts[:, 1],
                    z=inpaint_pts[:, 2],
                    mode="markers",
                    name=f"mask-depth inpaint ({len(inpaint_pts)})",
                    marker={"size": 2.4, "color": "rgba(245,158,11,0.9)"},
                    hoverinfo="skip",
                )
            )
        if len(prismatic_pts):
            traces.append(
                go.Scatter3d(
                    x=prismatic_pts[:, 0],
                    y=prismatic_pts[:, 1],
                    z=prismatic_pts[:, 2],
                    mode="markers",
                    name=f"prismatic side extrude ({len(prismatic_pts)})",
                    marker={"size": 1.8, "color": "rgba(6,182,212,0.55)"},
                    hoverinfo="skip",
                )
            )
        if len(graspgen_input_pts):
            traces.append(
                go.Scatter3d(
                    x=graspgen_input_pts[:, 0],
                    y=graspgen_input_pts[:, 1],
                    z=graspgen_input_pts[:, 2],
                    mode="markers",
                    name=f"GraspGen input hollow shell ({len(graspgen_input_pts)})",
                    marker={"size": 1.1, "color": "rgba(37,99,235,0.22)"},
                    hoverinfo="skip",
                    visible="legendonly",
                )
            )

        stage_colors = {
            "selected": "#dc2626",
            "height_rejected": "#2563eb",
            "workspace_rejected": "#f97316",
            "ik_rejected": "#f59e0b",
            "execution_failed": "#ec4899",
            "adapter_rejected": "#9333ea",
            "confidence_rejected": "#64748b",
            "collision_rejected": "#92400e",
        }
        for record in self.execution_debug_records:
            stage = "selected" if record.get("selected") else str(record.get("stage", ""))
            color = stage_colors.get(stage, "#94a3b8")
            points = []
            labels = []
            for label, key in (
                ("S", "start_contact_base"),
                ("A", "approach_contact_base"),
                ("P", "pregrasp_realign_contact_base"),
                ("G", "grasp_contact_base"),
                ("L", "lift_contact_base"),
            ):
                value = record.get(key)
                if isinstance(value, list) and len(value) == 3:
                    points.append([float(v) for v in value])
                    labels.append(f"{label}{record.get('index')}")
            if points:
                pts = np.asarray(points, dtype=np.float64)
                traces.append(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="lines+markers+text",
                        name=f"candidate {record.get('index')} {stage}",
                        text=labels,
                        textposition="top center",
                        line={"color": color, "width": 7 if record.get("selected") else 4},
                        marker={"size": 5 if record.get("selected") else 3.5, "color": color},
                        hovertext=[json.dumps(record, ensure_ascii=False, indent=2)] * len(points),
                        hoverinfo="text",
                    )
                )

        def add_pick_trace(
            *,
            name: str,
            key: str,
            color: str,
            symbol: str,
            marker_size: float,
            line_width: float,
        ) -> None:
            points = []
            labels = []
            hover = []
            for record in pick_records:
                point = record_xyz(record, key)
                if point is None:
                    continue
                phase = str(record.get("label", "?"))
                points.append(point)
                labels.append(phase)
                hover.append(json.dumps(record, ensure_ascii=False, indent=2))
            if not points:
                return
            pts = np.asarray(points, dtype=np.float64)
            traces.append(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines+markers+text",
                    name=name,
                    text=labels,
                    textposition="bottom center",
                    line={"color": color, "width": line_width},
                    marker={"size": marker_size, "color": color, "symbol": symbol},
                    hovertext=hover,
                    hoverinfo="text",
                )
            )

        add_pick_trace(
            name="planned contact target before realign",
            key="planned_contact_base",
            color="#dc2626",
            symbol="circle-open",
            marker_size=5.5,
            line_width=4,
        )
        add_pick_trace(
            name="post-realign commanded contact",
            key="commanded_pose_contact_base",
            color="#2563eb",
            symbol="diamond",
            marker_size=6.5,
            line_width=5,
        )
        add_pick_trace(
            name="actual TF contact",
            key="actual_contact_base",
            color="#111827",
            symbol="x",
            marker_size=6.5,
            line_width=5,
        )
        add_pick_trace(
            name="post-realign commanded EE origin",
            key="commanded",
            color="#0ea5e9",
            symbol="square-open",
            marker_size=4.5,
            line_width=2,
        )
        add_pick_trace(
            name="actual TF EE origin",
            key="actual",
            color="#7c3aed",
            symbol="square",
            marker_size=4.5,
            line_width=2,
        )

        # The approach-phase detection is the most trustworthy target observation:
        # the wrist camera is still ~8-13cm above the object, so the mask/depth is
        # clean. Grasp/close re-detections happen at very close range (finger
        # occlusion, depth dropout) and are frequently corrupted, so they are shown
        # de-emphasised and are excluded from the "reference target" alignment cue.
        reference_target: list[float] | None = None
        reliable_points = []
        reliable_labels = []
        reliable_hover = []
        suspect_points = []
        suspect_labels = []
        suspect_hover = []
        for record in pick_records:
            phase = str(record.get("label", "?"))
            hover = json.dumps(record, ensure_ascii=False, indent=2)
            observed_target = record_xyz(record, "observed_target_base")
            detected_target = target_detection_xyz(record)
            # approach-phase observation is the trusted reference for alignment
            if phase == "approach" and observed_target is not None and reference_target is None:
                reference_target = observed_target
            for kind, point in (("observed", observed_target), ("detected", detected_target)):
                if point is None:
                    continue
                if phase == "approach":
                    reliable_points.append(point)
                    reliable_labels.append(f"{kind} {phase}")
                    reliable_hover.append(hover)
                else:
                    suspect_points.append(point)
                    suspect_labels.append(f"{kind} {phase}")
                    suspect_hover.append(hover)

        if reliable_points:
            pts = np.asarray(reliable_points, dtype=np.float64)
            primary_centroid_label = "volume centroid" if self.args.centroid_source == "volume" else "surface centroid"
            traces.append(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers+text",
                    name=f"trusted target — {primary_centroid_label} (approach detection)",
                    text=reliable_labels,
                    textposition="top center",
                    marker={"size": 9, "color": "#e11d48", "symbol": "cross"},
                    hovertext=reliable_hover,
                    hoverinfo="text",
                )
            )
        if suspect_points:
            pts = np.asarray(suspect_points, dtype=np.float64)
            traces.append(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers+text",
                    name="close-range re-detection (unreliable)",
                    text=suspect_labels,
                    textposition="top center",
                    marker={"size": 4, "color": "rgba(225,29,72,0.35)", "symbol": "cross"},
                    hovertext=suspect_hover,
                    hoverinfo="text",
                )
            )

        # Show the alternative centroid (the one not selected by --centroid-source)
        # so the operator can visually compare surface vs volume centroid placement.
        if self.observed_target_base_alt is not None:
            alt = self.observed_target_base_alt
            alt_label = "surface centroid" if self.args.centroid_source == "volume" else "volume centroid"
            alt_color = "#8b5cf6"  # purple
            # primary centroid (used for ranking) already shown via trusted target trace;
            # add the alternative as a distinct marker + connecting line.
            traces.append(
                go.Scatter3d(
                    x=[alt[0]],
                    y=[alt[1]],
                    z=[alt[2]],
                    mode="markers+text",
                    name=f"alt target ({alt_label})",
                    text=[alt_label],
                    textposition="bottom center",
                    marker={"size": 8, "color": alt_color, "symbol": "diamond"},
                    hoverinfo="name",
                )
            )
            if reference_target is not None:
                seg = np.asarray([reference_target, alt], dtype=np.float64)
                delta = np.linalg.norm(
                    np.asarray(reference_target, dtype=np.float64) - np.asarray(alt, dtype=np.float64)
                )
                traces.append(
                    go.Scatter3d(
                        x=seg[:, 0],
                        y=seg[:, 1],
                        z=seg[:, 2],
                        mode="lines",
                        name=f"centroid Δ ({delta * 100:.1f} cm)",
                        line={"color": alt_color, "width": 4, "dash": "dot"},
                        hoverinfo="name",
                    )
                )

        # Draw the alignment gap: trusted target vs the planned/actual grasp contact.
        # A visible vertical (mostly z) segment here is the real "gripper stops above
        # the object" error the operator observes on the robot.
        grasp_contact = None
        for record in self.execution_debug_records:
            if record.get("selected"):
                value = record.get("grasp_contact_base")
                if isinstance(value, list) and len(value) == 3:
                    grasp_contact = [float(v) for v in value]
                break
        if reference_target is not None and grasp_contact is not None:
            gap = np.asarray(reference_target, dtype=np.float64) - np.asarray(grasp_contact, dtype=np.float64)
            gap_norm = float(np.linalg.norm(gap))
            seg = np.asarray([reference_target, grasp_contact], dtype=np.float64)
            traces.append(
                go.Scatter3d(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    z=seg[:, 2],
                    mode="lines+text",
                    name=f"target-vs-grasp gap ({gap_norm * 100:.1f} cm, dz={gap[2] * 100:.1f} cm)",
                    text=["target", "grasp contact"],
                    textposition="middle right",
                    line={"color": "#f59e0b", "width": 6, "dash": "dash"},
                    hoverinfo="name",
                )
            )

        for label, color, lines in selected_gripper_lines:
            for i, line in enumerate(lines):
                traces.append(
                    self._plotly_line_trace(
                        line,
                        name=label,
                        color=color,
                        width=8 if "grasp" in label else 5,
                        showlegend=i == 0,
                    )
                )

        for label, color, meshes in selected_gripper_meshes:
            for i, (part_name, vertices, faces) in enumerate(meshes):
                traces.append(
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        name=f"{label}: {part_name}",
                        color=color,
                        opacity=0.28 if "start" in label else 0.42,
                        flatshading=True,
                        lighting={"ambient": 0.45, "diffuse": 0.75, "specular": 0.2, "roughness": 0.8},
                        showlegend=i == 0,
                        hoverinfo="name",
                    )
                )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=(
                "SO101 execution path (object/scene cloud projected with START-time TF x hand-eye). "
                "Dashed amber = trusted-target vs grasp-contact gap"
            ),
            scene={
                "xaxis_title": f"X {self.args.base_frame} (m)",
                "yaxis_title": f"Y {self.args.base_frame} (m)",
                "zaxis_title": f"Z {self.args.base_frame} (m)",
                "xaxis": {"range": [float(roi_min[0]), float(roi_max[0])]},
                "yaxis": {"range": [float(roi_min[1]), float(roi_max[1])]},
                "zaxis": {"range": [float(roi_min[2]), float(roi_max[2])]},
                "aspectmode": "data",
                "camera": {"eye": {"x": 1.1, "y": -1.5, "z": 0.9}, "up": {"x": 0, "y": 0, "z": 1}},
            },
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
            showlegend=True,
        )
        fig.write_html(str(path), include_plotlyjs=True, full_html=True, config={"displaylogo": False})

    def _render_execution_debug_preview_svg(self, path: Path) -> None:
        color_by_stage = {
            "selected": "#dc2626",
            "ik_pass": "#16a34a",
            "ik_rejected": "#f59e0b",
            "workspace_rejected": "#9ca3af",
            "height_rejected": "#9ca3af",
            "confidence_rejected": "#9ca3af",
            "collision_rejected": "#9ca3af",
            "execution_failed": "#ec4899",
            "adapter_rejected": "#9ca3af",
        }
        paths: list[tuple[dict[str, object], list[list[float]]]] = []
        all_points: list[list[float]] = []
        for record in self.execution_debug_records:
            path_points = []
            for key in (
                "approach_contact_base",
                "pregrasp_realign_contact_base",
                "grasp_contact_base",
                "lift_contact_base",
            ):
                value = record.get(key)
                if isinstance(value, list) and len(value) == 3:
                    point = [float(v) for v in value]
                    path_points.append(point)
                    all_points.append(point)
            grasp = record.get("grasp")
            if isinstance(grasp, list) and len(grasp) == 3:
                all_points.append([float(v) for v in grasp])
            if path_points:
                paths.append((record, path_points))
        if not all_points:
            return

        pts = np.array(all_points, dtype=np.float64)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        mins[2] = min(float(mins[2]), float(self.args.min_contact_z))
        maxs[2] = max(float(maxs[2]), float(self.args.min_contact_z))
        pad = np.maximum((maxs - mins) * 0.12, 0.02)
        mins -= pad
        maxs += pad
        width = 1265
        height = 760
        panel_w = 560
        panel_h = 520
        margin_x = 45
        margin_y = 125

        def project(point: list[float], panel: int) -> tuple[float, float]:
            x = point[0]
            y_or_z = point[1] if panel == 0 else point[2]
            min_yz = mins[1] if panel == 0 else mins[2]
            max_yz = maxs[1] if panel == 0 else maxs[2]
            px = margin_x + panel * (panel_w + 55) + (x - mins[0]) / max(maxs[0] - mins[0], 1e-6) * panel_w
            py = margin_y + panel_h - (y_or_z - min_yz) / max(max_yz - min_yz, 1e-6) * panel_h
            return px, py

        def fmt(value: object) -> str:
            if isinstance(value, float | int):
                return f"{float(value):.4f}"
            return "n/a"

        selected = next((record for record in self.execution_debug_records if record.get("selected")), None)
        selected_summary = "no selected candidate"
        if selected is not None:
            grasp_contact = selected.get("grasp_contact_base")
            z_text = "n/a"
            if isinstance(grasp_contact, list) and len(grasp_contact) == 3:
                z_text = f"{float(grasp_contact[2]):.4f}"
            selected_summary = (
                f"selected idx={selected.get('index')} stage={selected.get('stage')} "
                f"conf={fmt(selected.get('confidence'))} grasp_contact_z={z_text} "
                f"min_contact_z={self.args.min_contact_z:.4f}"
            )

        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="45" y="36" font-size="22" font-family="monospace">SO101 execution contact-path diagnostic</text>',
            '<text x="45" y="63" font-size="13" font-family="monospace">Top view shows where the SO101 contact point moves on the table. Side view shows height vs min_contact_z.</text>',
            f'<text x="45" y="88" font-size="13" font-family="monospace" fill="#0f172a">{selected_summary}</text>',
            '<text x="45" y="710" font-size="13" font-family="monospace" fill="#9ca3af">gray: rejected before IK or guard</text>',
            '<text x="310" y="710" font-size="13" font-family="monospace" fill="#f59e0b">yellow: IK rejected</text>',
            '<text x="520" y="710" font-size="13" font-family="monospace" fill="#16a34a">green: IK passed</text>',
            '<text x="730" y="710" font-size="13" font-family="monospace" fill="#dc2626">red: selected</text>',
            '<text x="45" y="735" font-size="12" font-family="monospace" fill="#475569">A=approach, P=pregrasp realign, G=grasp, L=lift contact. Dashed line is min_contact_z.</text>',
        ]
        for panel, title in enumerate(("Top view: base X-Y contact path", "Side view: base X-Z contact height")):
            x0 = margin_x + panel * (panel_w + 55)
            elements.append(
                f'<rect x="{x0}" y="{margin_y}" width="{panel_w}" height="{panel_h}" fill="#f8fafc" stroke="#cbd5e1"/>'
            )
            elements.append(f'<text x="{x0}" y="{margin_y - 14}" font-size="16" font-family="monospace">{title}</text>')
            elements.append(
                f'<line x1="{x0}" y1="{margin_y + panel_h}" x2="{x0 + panel_w}" y2="{margin_y + panel_h}" stroke="#64748b"/>'
            )
            elements.append(f'<line x1="{x0}" y1="{margin_y}" x2="{x0}" y2="{margin_y + panel_h}" stroke="#64748b"/>')
            y_axis = "Y" if panel == 0 else "Z"
            elements.append(
                f'<text x="{x0 + panel_w - 45}" y="{margin_y + panel_h + 22}" font-size="12" font-family="monospace">X (m)</text>'
            )
            elements.append(
                f'<text x="{x0 + 6}" y="{margin_y + 16}" font-size="12" font-family="monospace">{y_axis} (m)</text>'
            )
        min_z_left = project([float(mins[0]), 0.0, float(self.args.min_contact_z)], 1)
        min_z_right = project([float(maxs[0]), 0.0, float(self.args.min_contact_z)], 1)
        elements.append(
            f'<line x1="{min_z_left[0]:.1f}" y1="{min_z_left[1]:.1f}" x2="{min_z_right[0]:.1f}" y2="{min_z_right[1]:.1f}" stroke="#ef4444" stroke-width="2" stroke-dasharray="7 5"/>'
        )
        elements.append(
            f'<text x="{min_z_left[0] + 8:.1f}" y="{min_z_left[1] - 8:.1f}" font-size="12" font-family="monospace" fill="#ef4444">min_contact_z={self.args.min_contact_z:.4f}</text>'
        )
        for record, path_points in paths:
            stage = "selected" if record.get("selected") else str(record.get("stage", ""))
            color = color_by_stage.get(stage, "#9ca3af")
            stroke_width = 4 if record.get("selected") else 2
            for panel in (0, 1):
                coords = [project(point, panel) for point in path_points]
                polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
                elements.append(
                    f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="{stroke_width}" opacity="0.9"/>'
                )
                for label, (gx, gy) in zip(("A", "P", "G", "L"), coords, strict=False):
                    radius = 7 if label == "G" and record.get("selected") else 5
                    elements.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="{radius}" fill="{color}" stroke="white"/>')
                    elements.append(
                        f'<text x="{gx + 7:.1f}" y="{gy - 7:.1f}" font-size="11" font-family="monospace" fill="#0f172a">{label}{record.get("index", "")}</text>'
                    )
        elements.append("</svg>")
        path.write_text("\n".join(elements), encoding="utf-8")

    def _render_execution_debug_preview(self, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        color_by_stage = {
            "selected": "#dc2626",
            "ik_pass": "#16a34a",
            "ik_rejected": "#f59e0b",
            "workspace_rejected": "#9ca3af",
            "height_rejected": "#9ca3af",
            "confidence_rejected": "#9ca3af",
            "collision_rejected": "#9ca3af",
            "execution_failed": "#ec4899",
            "adapter_rejected": "#9ca3af",
        }
        fig = plt.figure(figsize=(11, 8), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        points = []
        line_segments = []
        line_colors = []
        for record in self.execution_debug_records:
            stage = "selected" if record.get("selected") else str(record.get("stage", ""))
            color = color_by_stage.get(stage, "#9ca3af")
            path_points = []
            for key in (
                "approach_contact_base",
                "pregrasp_realign_contact_base",
                "grasp_contact_base",
                "lift_contact_base",
            ):
                value = record.get(key)
                if isinstance(value, list) and len(value) == 3:
                    pt = [float(v) for v in value]
                    path_points.append(pt)
                    points.append(pt)
            if len(path_points) >= 2:
                line_segments.append(path_points)
                line_colors.append(color)
            grasp = record.get("grasp")
            if isinstance(grasp, list) and len(grasp) == 3:
                points.append([float(v) for v in grasp])
                ax.scatter(
                    float(grasp[0]),
                    float(grasp[1]),
                    float(grasp[2]),
                    color=color,
                    s=70 if record.get("selected") else 34,
                    depthshade=False,
                )
                ax.text(float(grasp[0]), float(grasp[1]), float(grasp[2]), str(record.get("index", "")), fontsize=8)
        if line_segments:
            ax.add_collection3d(Line3DCollection(line_segments, colors=line_colors, linewidths=2.2, alpha=0.9))
        if points:
            pts = np.array(points, dtype=np.float64)
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            center = (mins + maxs) / 2.0
            span = max(float(np.max(maxs - mins)) / 2.0, 0.05)
            ax.set_xlim(center[0] - span, center[0] + span)
            ax.set_ylim(center[1] - span, center[1] + span)
            ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_xlabel(f"X {self.args.base_frame} (m)")
        ax.set_ylabel(f"Y {self.args.base_frame} (m)")
        ax.set_zlabel(f"Z {self.args.base_frame} (m)")
        ax.set_title("SO101 execution-stage candidate paths (contact center)")
        handles = [
            plt.Line2D([0], [0], color="#9ca3af", lw=3, label="rejected before IK"),
            plt.Line2D([0], [0], color="#f59e0b", lw=3, label="IK rejected"),
            plt.Line2D([0], [0], color="#16a34a", lw=3, label="IK passed"),
            plt.Line2D([0], [0], color="#dc2626", lw=3, label="selected"),
        ]
        ax.legend(handles=handles, loc="upper left")
        ax.view_init(elev=24, azim=-55)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

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
                use_vol = self.args.centroid_source == "volume"
                src = detection.volume_centroid_xyz if use_vol else detection.centroid_xyz
                centroid = np.array(
                    [src.x, src.y, src.z],
                    dtype=np.float64,
                )
                self.observed_target_base = self.detection_to_base(detection, base_to_gripper_tf)
                # Also compute the alternative centroid in base frame for the HTML preview.
                self.observed_target_base_alt = self.detection_to_base(
                    detection, base_to_gripper_tf, use_volume=not use_vol
                )
                centroid_reason = f"ok({self.args.centroid_source})"
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

    def select_graspgen_candidates(
        self,
        base_to_gripper_tf,
    ) -> list[dict[str, object]]:
        self.wait_ik_ready()
        t_base_gripper_start = transform_to_matrix(base_to_gripper_tf)
        start, start_quat = pose_from_matrix(t_base_gripper_start)
        self.last_t_base_camera = t_base_gripper_start @ np.array(self.handeye_matrix, dtype=np.float64)
        candidates = self.request_graspgen_candidates(base_to_gripper_tf)
        self.execution_debug_records = []
        max_candidates = int(self.args.max_candidates)
        max_count = len(candidates) if max_candidates <= 0 else min(len(candidates), max(1, max_candidates))
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
                f"fixed_finger_margin={self.args.target_fixed_finger_margin_m:.4f} "
                f"fixed_finger_margin_max={self.args.target_fixed_finger_margin_max_m:.4f} "
                f"fixed_finger_margin_width_ref={self.args.target_fixed_finger_margin_width_ref_m:.4f} "
                f"fixed_finger_margin_width_gain={self.args.target_fixed_finger_margin_width_gain:.3f} "
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

        accepted: list[dict[str, object]] = []
        for index, candidate, centroid_dist_camera, topdown_score in ranked_candidates:
            confidence = float(candidate.confidence)
            if confidence < self.args.min_grasp_confidence:
                self._record_execution_candidate(
                    index=index,
                    confidence=confidence,
                    stage="confidence_rejected",
                    reason=f"confidence_below_{self.args.min_grasp_confidence:.3f}",
                    collision_free=bool(candidate.collision_free),
                    topdown_score=topdown_score,
                    centroid_dist_camera=centroid_dist_camera,
                )
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} conf={confidence:.3f} "
                    f"reason=confidence_below_{self.args.min_grasp_confidence:.3f}",
                    flush=True,
                )
                continue
            if self.args.require_collision_free_grasp and not bool(candidate.collision_free):
                self._record_execution_candidate(
                    index=index,
                    confidence=confidence,
                    stage="collision_rejected",
                    reason="not_collision_free",
                    collision_free=bool(candidate.collision_free),
                    topdown_score=topdown_score,
                    centroid_dist_camera=centroid_dist_camera,
                )
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
                self._record_execution_candidate(
                    index=index,
                    confidence=confidence,
                    stage="adapter_rejected",
                    reason=str(exc),
                    collision_free=bool(candidate.collision_free),
                    topdown_score=topdown_score,
                    centroid_dist_camera=centroid_dist_camera,
                )
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} conf={confidence:.3f} reason={exc}",
                    flush=True,
                )
                continue
            approach, grasp, lift, quat, radius = self.graspgen_targets_from_pose(t_base_ee, t_base_graspgen)
            contact = self.graspgen_contact_point_base(t_base_graspgen)
            execution_contact = self._contact_for_pose(grasp, quat, target_contact)
            width, width_quality = candidate_target_width(candidate)
            grasp_mesh_min_z = self._so101_gripper_mesh_min_z(grasp, quat, width)
            so101_tabletop_clearance = self._so101_tabletop_clearance(approach, grasp, quat, width)
            workspace_ok, workspace_reason = self._is_within_workspace(grasp, radius)
            height_ok, height_reason = self._graspgen_height_guard(
                approach,
                grasp,
                execution_contact,
            )
            grasp_mesh_min_z_text = f"{grasp_mesh_min_z:.4f}" if grasp_mesh_min_z is not None else "n/a"
            so101_tabletop_clearance_text = (
                f"{so101_tabletop_clearance:.4f}" if so101_tabletop_clearance is not None else "n/a"
            )
            camera_xyz = tuple(float(v) for v in t_camera_graspgen[:3, 3])
            camera_contact = self.graspgen_contact_point_camera(t_camera_graspgen)
            graspgen_axis_z = tuple(float(v) for v in t_base_graspgen[:3, 2])
            centroid_dist_text = (
                f" centroid_dist_camera={centroid_dist_camera:.4f}" if centroid_dist_camera is not None else ""
            )
            print(
                f"GRASPGEN_CANDIDATE idx={index} conf={confidence:.3f} collision_free={bool(candidate.collision_free)} "
                f"graspgen_origin_camera={fmt_xyz(camera_xyz)} target_ee_grasp={fmt_xyz(grasp)} "
                f"contact_camera={fmt_xyz(camera_contact)} contact_base={fmt_xyz(contact)} "
                f"execution_contact_base={fmt_xyz(execution_contact)} "
                f"so101_mesh_min_z={grasp_mesh_min_z_text} "
                f"so101_tabletop_clearance={so101_tabletop_clearance_text} "
                f"approach_axis_base={fmt_xyz(graspgen_axis_z)} target_ee_quat={fmt_quat(quat)} "
                f"topdown_score={topdown_score:.3f} "
                f"target_width={width:.4f} width_quality={width_quality:.3f} width_comp={width_reason} "
                f"target_contact={fmt_xyz(target_contact)} adapter_xyz={fmt_xyz(adapter_xyz)} "
                f"radius={radius:.4f}{centroid_dist_text}",
                flush=True,
            )
            if not workspace_ok:
                self._record_execution_candidate(
                    index=index,
                    confidence=confidence,
                    stage="workspace_rejected",
                    reason=workspace_reason,
                    collision_free=bool(candidate.collision_free),
                    topdown_score=topdown_score,
                    centroid_dist_camera=centroid_dist_camera,
                    approach=approach,
                    grasp=grasp,
                    lift=lift,
                    quat=quat,
                    target_contact=target_contact,
                    target_width_m=width,
                    target_width_quality=width_quality,
                    grasp_mesh_min_z=grasp_mesh_min_z,
                    so101_tabletop_clearance_m=so101_tabletop_clearance,
                    adapter_xyz=adapter_xyz,
                    width_reason=width_reason,
                )
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} "
                    f"radius={radius:.4f} reason={workspace_reason}",
                    flush=True,
                )
                continue
            if not height_ok:
                self._record_execution_candidate(
                    index=index,
                    confidence=confidence,
                    stage="height_rejected",
                    reason=height_reason,
                    collision_free=bool(candidate.collision_free),
                    topdown_score=topdown_score,
                    centroid_dist_camera=centroid_dist_camera,
                    approach=approach,
                    grasp=grasp,
                    lift=lift,
                    quat=quat,
                    target_contact=target_contact,
                    target_width_m=width,
                    target_width_quality=width_quality,
                    grasp_mesh_min_z=grasp_mesh_min_z,
                    so101_tabletop_clearance_m=so101_tabletop_clearance,
                    adapter_xyz=adapter_xyz,
                    width_reason=width_reason,
                )
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} "
                    f"contact={fmt_xyz(execution_contact)} reason=height_guard_failed {height_reason}",
                    flush=True,
                )
                continue
            if so101_tabletop_clearance is not None and so101_tabletop_clearance < float(
                self.args.so101_tabletop_clearance
            ):
                reason = (
                    f"so101_tabletop_clearance {so101_tabletop_clearance:.3f} "
                    f"< {float(self.args.so101_tabletop_clearance):.3f}"
                )
                self._record_execution_candidate(
                    index=index,
                    confidence=confidence,
                    stage="collision_rejected",
                    reason=reason,
                    collision_free=bool(candidate.collision_free),
                    topdown_score=topdown_score,
                    centroid_dist_camera=centroid_dist_camera,
                    approach=approach,
                    grasp=grasp,
                    lift=lift,
                    quat=quat,
                    target_contact=target_contact,
                    target_width_m=width,
                    target_width_quality=width_quality,
                    grasp_mesh_min_z=grasp_mesh_min_z,
                    so101_tabletop_clearance_m=so101_tabletop_clearance,
                    adapter_xyz=adapter_xyz,
                    width_reason=width_reason,
                )
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} "
                    f"reason=so101_tabletop_failed {reason}",
                    flush=True,
                )
                continue

            checks = [("approach", approach)]
            if self.args.require_grasp_ik:
                checks.append(("grasp", grasp))
            if self.args.require_lift_ik:
                checks.append(("lift", lift))

            failed = False
            failed_reason = ""
            for label, xyz in checks:
                ik_quat = quat if self.args.ik_check_orientation else None
                ik_ok, code = self.check_ik(f"graspgen_{index}_{label}", xyz, ik_quat)
                if not ik_ok:
                    failed_reason = f"ik_failed_{label} code={code}"
                    print(
                        f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} "
                        f"reason=ik_failed_{label} code={code}",
                        flush=True,
                    )
                    failed = True
                    break
            if failed:
                self._record_execution_candidate(
                    index=index,
                    confidence=confidence,
                    stage="ik_rejected",
                    reason=failed_reason,
                    collision_free=bool(candidate.collision_free),
                    topdown_score=topdown_score,
                    centroid_dist_camera=centroid_dist_camera,
                    approach=approach,
                    grasp=grasp,
                    lift=lift,
                    quat=quat,
                    target_contact=target_contact,
                    target_width_m=width,
                    target_width_quality=width_quality,
                    grasp_mesh_min_z=grasp_mesh_min_z,
                    so101_tabletop_clearance_m=so101_tabletop_clearance,
                    adapter_xyz=adapter_xyz,
                    width_reason=width_reason,
                )
                continue

            print(
                f"GRASPGEN_CANDIDATE_ACCEPT idx={index} conf={confidence:.3f} "
                f"approach={fmt_xyz(approach)} grasp={fmt_xyz(grasp)} lift={fmt_xyz(lift)} "
                f"quat={fmt_quat(quat)}",
                flush=True,
            )
            self._record_execution_candidate(
                index=index,
                confidence=confidence,
                stage="ik_pass",
                reason="passed_workspace_height_tabletop_ik",
                collision_free=bool(candidate.collision_free),
                topdown_score=topdown_score,
                centroid_dist_camera=centroid_dist_camera,
                approach=approach,
                grasp=grasp,
                lift=lift,
                quat=quat,
                start=start,
                start_quat=start_quat,
                target_contact=target_contact,
                target_width_m=width,
                target_width_quality=width_quality,
                grasp_mesh_min_z=grasp_mesh_min_z,
                so101_tabletop_clearance_m=so101_tabletop_clearance,
                adapter_xyz=adapter_xyz,
                width_reason=width_reason,
                selected=False,
            )
            accepted.append(
                {
                    "index": index,
                    "approach": approach,
                    "grasp": grasp,
                    "lift": lift,
                    "quat": quat,
                    "radius": radius,
                    "target_contact": tuple(float(v) for v in target_contact),
                    "plan_contact": contact,
                }
            )

        self._write_execution_debug_outputs()
        if accepted:
            return accepted
        raise RuntimeError("No GraspGen candidate passed workspace, height, and IK filters")

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
        candidate = self.select_graspgen_candidates(base_to_gripper_tf)[0]
        index = int(candidate["index"])
        self.mark_execution_candidate_attempt(
            index,
            selected=True,
            stage="selected",
            reason="selected_first_candidate_passing_workspace_height_ik",
        )
        self.selected_target_contact_ee = tuple(float(v) for v in candidate["target_contact"])
        self.selected_plan_contact_base = tuple(float(v) for v in candidate["plan_contact"])
        self.current_execution_candidate_index = index
        return (
            candidate["approach"],
            candidate["grasp"],
            candidate["lift"],
            candidate["quat"],
            float(candidate["radius"]),
        )

    def detection_to_base(
        self, detection: Detection2D, base_to_gripper_tf, use_volume: bool | None = None
    ) -> tuple[float, float, float]:
        if use_volume is None:
            use_volume = self.args.centroid_source == "volume"
        src = detection.volume_centroid_xyz if use_volume else detection.centroid_xyz
        camera_point = (src.x, src.y, src.z)
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
                if self.args.max_candidates > 0 and len(candidates) >= self.args.max_candidates:
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

    def pregrasp_realign_pose(
        self,
        grasp_xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float] | None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
        grasp_contact = self.planned_contact_for_pose(grasp_xyz, quat_xyzw)
        reference_z = grasp_contact[2]
        reference_source = "target_contact"
        if quat_xyzw is not None:
            selected = next((record for record in self.execution_debug_records if record.get("selected")), None)
            width = None
            if selected is not None and isinstance(selected.get("target_width_m"), float | int):
                width = float(selected["target_width_m"])
            mesh_min_z = self._so101_gripper_mesh_min_z(grasp_xyz, quat_xyzw, width)
            if mesh_min_z is not None:
                reference_z = mesh_min_z
                reference_source = "so101_mesh_min_z"
        object_top_z, top_source = self.object_top_z_base()
        clearance = max(0.0, float(self.args.pregrasp_realign_clearance))
        if object_top_z is None:
            target_reference_z = reference_z + clearance
        else:
            target_reference_z = object_top_z + clearance
        dz = target_reference_z - reference_z
        pregrasp = (grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + dz)
        pregrasp_contact = self.planned_contact_for_pose(pregrasp, quat_xyzw)
        print(
            f"PREGRASP_REALIGN target_reference_z={target_reference_z:.4f} clearance={clearance:.4f} "
            f"reference_z={reference_z:.4f} reference_source={reference_source} "
            f"object_top_z={'n/a' if object_top_z is None else f'{object_top_z:.4f}'} "
            f"top_source={top_source} grasp_contact={fmt_xyz(grasp_contact)} "
            f"pregrasp={fmt_xyz(pregrasp)} pregrasp_contact={fmt_xyz(pregrasp_contact)}",
            flush=True,
        )
        return pregrasp, pregrasp_contact, f"{top_source}:{reference_source}"

    def apply_realign_delta_to_descent(
        self,
        original_pregrasp: tuple[float, float, float],
        realigned_pregrasp: tuple[float, float, float],
        grasp: tuple[float, float, float],
        lift: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        delta = sub_xyz(realigned_pregrasp, original_pregrasp)
        # Pregrasp is a safe high pose; only XY realignment should affect the final descent depth.
        descent_delta = (delta[0], delta[1], 0.0)
        corrected_grasp = add_xyz(grasp, descent_delta)
        corrected_lift = (corrected_grasp[0], corrected_grasp[1], corrected_grasp[2] + self.args.final_lift)
        print(
            f"PREGRASP_REALIGN_APPLY pregrasp_delta={fmt_xyz(delta)} "
            f"descent_delta={fmt_xyz(descent_delta)} ignored_z_delta={delta[2]:.4f} "
            f"corrected_grasp={fmt_xyz(corrected_grasp)} corrected_lift={fmt_xyz(corrected_lift)}",
            flush=True,
        )
        _ = lift
        return corrected_grasp, corrected_lift

    def log_grasp_contact_residual(
        self,
        grasp: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float] | None,
    ) -> None:
        if quat_xyzw is None:
            return
        planned_contact_base = self.planned_contact_for_pose(grasp, quat_xyzw)
        self.current_plan_contact_base = planned_contact_base
        self.realign_target_contact_base_by_phase["grasp"] = planned_contact_base
        _, correction, error_norm = self.correction_for_contact_alignment(grasp, planned_contact_base)
        xy_error = math.hypot(correction[0], correction[1])
        warn_xy = max(0.0, float(self.args.grasp_realign_max_xy_error))
        realign_xy = float(self.args.grasp_residual_realign_xy_error)
        abort_xy = float(self.args.grasp_residual_abort_xy_error)
        action = "continue_without_low_height_xy_realign"
        if warn_xy > 0.0 and xy_error > warn_xy:
            action = "warn_continue"
        print(
            f"CONTACT_REALIGN_CHECK phase=grasp error={fmt_xyz(correction)} "
            f"xy_error={xy_error:.4f} error_norm={error_norm:.4f} warn_xy={warn_xy:.4f} "
            f"realign_xy={realign_xy:.4f} abort_xy={abort_xy:.4f} "
            f"planned_contact={fmt_xyz(planned_contact_base)} action={action}",
            flush=True,
        )

    def update_selected_execution_pose(
        self,
        *,
        approach: tuple[float, float, float] | None = None,
        pregrasp: tuple[float, float, float] | None = None,
        grasp: tuple[float, float, float] | None = None,
        lift: tuple[float, float, float] | None = None,
        quat_xyzw: tuple[float, float, float, float] | None = None,
        pregrasp_top_source: str | None = None,
    ) -> None:
        selected = next((record for record in self.execution_debug_records if record.get("selected")), None)
        if selected is None:
            return
        contact = selected.get("target_contact_ee")
        contact_tuple = tuple(float(v) for v in contact) if isinstance(contact, list) and len(contact) == 3 else None
        quat = quat_xyzw
        if quat is None:
            existing_quat = selected.get("quat_xyzw")
            quat = (
                tuple(float(v) for v in existing_quat)
                if isinstance(existing_quat, list) and len(existing_quat) == 4
                else None
            )

        if approach is not None:
            selected["approach"] = _json_xyz(approach)
            if contact_tuple is not None and quat is not None:
                selected["approach_contact_base"] = _json_xyz(self._contact_for_pose(approach, quat, contact_tuple))
        if pregrasp is not None:
            selected["pregrasp_realign"] = _json_xyz(pregrasp)
            if pregrasp_top_source:
                selected["pregrasp_realign_top_source"] = pregrasp_top_source
            if contact_tuple is not None and quat is not None:
                selected["pregrasp_realign_contact_base"] = _json_xyz(
                    self._contact_for_pose(pregrasp, quat, contact_tuple)
                )
        if grasp is not None:
            selected["grasp"] = _json_xyz(grasp)
            if contact_tuple is not None and quat is not None:
                selected["grasp_contact_base"] = _json_xyz(self._contact_for_pose(grasp, quat, contact_tuple))
        if lift is not None:
            selected["lift"] = _json_xyz(lift)
            if contact_tuple is not None and quat is not None:
                selected["lift_contact_base"] = _json_xyz(self._contact_for_pose(lift, quat, contact_tuple))

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
        self.realign_target_contact_base_by_phase[phase] = planned_contact_base
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
        record: dict[str, object] = {
            "label": label,
            "commanded": _json_xyz(commanded_xyz),
            "actual": _json_xyz(actual_xyz),
            "actual_minus_command": _json_xyz(pose_delta),
            "actual_minus_command_norm": round(norm_xyz(pose_delta), 6),
            "actual_quat_xyzw": _json_quat(actual_quat),
        }
        if self.current_execution_candidate_index is not None:
            record["candidate_index"] = int(self.current_execution_candidate_index)
        rot_delta_text = ""
        if commanded_quat_xyzw is not None:
            rot_delta = quat_delta_deg(actual_quat, commanded_quat_xyzw)
            record["commanded_quat_xyzw"] = _json_quat(commanded_quat_xyzw)
            record["rot_delta_deg"] = round(rot_delta, 6)
            rot_delta_text = f" commanded_q={fmt_quat(commanded_quat_xyzw)} rot_delta_deg={rot_delta:.2f}"
        print(
            f"PICK_DIAG_POSE label={label} commanded={fmt_xyz(commanded_xyz)} actual={fmt_xyz(actual_xyz)} "
            f"actual_minus_command={fmt_xyz(pose_delta)} norm={norm_xyz(pose_delta):.4f} "
            f"actual_q={fmt_quat(actual_quat)}{rot_delta_text}",
            flush=True,
        )

        contact_ee = self.selected_target_contact_ee or tuple(float(v) for v in _static_target_contact(self.args))
        actual_contact = self.contact_base_from_tf(base_to_gripper_tf, contact_ee)
        commanded_pose_contact = self.planned_contact_for_pose(commanded_xyz, commanded_quat_xyzw)
        planned_contact = self.realign_target_contact_base_by_phase.get(label)
        planned_contact_source = "contact_realign_target" if planned_contact is not None else "commanded_pose_contact"
        if planned_contact is None and label == "close":
            planned_contact = self.realign_target_contact_base_by_phase.get("grasp")
            if planned_contact is not None:
                planned_contact_source = "contact_realign_target:grasp"
        if planned_contact is None:
            planned_contact = commanded_pose_contact
        self.current_plan_contact_base = planned_contact
        record["contact_ee"] = _json_xyz(contact_ee)
        record["actual_contact_base"] = _json_xyz(actual_contact)
        record["commanded_pose_contact_base"] = _json_xyz(commanded_pose_contact)
        commanded_pose_contact_delta = sub_xyz(actual_contact, commanded_pose_contact)
        record["actual_minus_commanded_pose_contact"] = _json_xyz(commanded_pose_contact_delta)
        record["actual_minus_commanded_pose_contact_norm"] = round(norm_xyz(commanded_pose_contact_delta), 6)
        if planned_contact is not None:
            contact_delta = sub_xyz(actual_contact, planned_contact)
            record["planned_contact_base"] = _json_xyz(planned_contact)
            record["planned_contact_source"] = planned_contact_source
            record["actual_minus_planned_contact"] = _json_xyz(contact_delta)
            record["actual_minus_planned_contact_norm"] = round(norm_xyz(contact_delta), 6)
            print(
                f"PICK_DIAG_CONTACT label={label} contact_ee={fmt_xyz(contact_ee)} "
                f"planned={fmt_xyz(planned_contact)} source={planned_contact_source} actual={fmt_xyz(actual_contact)} "
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
            record["observed_target_base"] = _json_xyz(self.observed_target_base)
            record["observed_minus_gripper"] = _json_xyz(observed_target_minus_gripper)
            record["observed_minus_gripper_norm"] = round(norm_xyz(observed_target_minus_gripper), 6)
            record["observed_minus_contact"] = _json_xyz(observed_target_minus_contact)
            record["observed_minus_contact_norm"] = round(norm_xyz(observed_target_minus_contact), 6)
            print(
                f"PICK_DIAG_OBS_TARGET label={label} observed_target_base={fmt_xyz(self.observed_target_base)} "
                f"observed_minus_gripper={fmt_xyz(observed_target_minus_gripper)} "
                f"gripper_norm={norm_xyz(observed_target_minus_gripper):.4f} "
                f"observed_minus_contact={fmt_xyz(observed_target_minus_contact)} "
                f"contact_norm={norm_xyz(observed_target_minus_contact):.4f}",
                flush=True,
            )

        if not detect_target or not self.args.pick_diagnostics_detect:
            self.pick_diagnostic_records.append(record)
            return

        try:
            detection = self.detect_target()
            target_base = self.detection_to_base(detection, base_to_gripper_tf)
        except Exception as exc:
            record["target_detection"] = {"success": False, "error": str(exc)}
            self.pick_diagnostic_records.append(record)
            print(f"PICK_DIAG_TARGET label={label} success=False error={exc}", flush=True)
            return

        target_minus_gripper = sub_xyz(target_base, actual_xyz)
        target_minus_contact = sub_xyz(target_base, actual_contact)
        self.observed_target_base = target_base
        record["target_detection"] = {
            "success": True,
            "target_base": _json_xyz(target_base),
            "target_minus_gripper": _json_xyz(target_minus_gripper),
            "target_minus_gripper_norm": round(norm_xyz(target_minus_gripper), 6),
            "target_minus_contact": _json_xyz(target_minus_contact),
            "target_minus_contact_norm": round(norm_xyz(target_minus_contact), 6),
        }
        self.pick_diagnostic_records.append(record)
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
                raise PickExecutionError("Pick task failed during prepare", phase="prepare", retryable=True)

            ok = self.run_task(
                f"{task_id}_approach",
                f"{task_desc}: approach",
                [make_move_step("move_above_target", approach, self.args.approach_speed, move_quat)],
            )
            if not ok:
                raise PickExecutionError("Pick task failed during approach", phase="approach", retryable=True)
            approach = self.realign_contact("approach", approach, self.args.approach_speed, move_quat)
            self.update_selected_execution_pose(approach=approach, quat_xyzw=move_quat)
            self.sample_pick_diagnostics(
                "approach",
                approach,
                commanded_quat_xyzw=move_quat,
                detect_target=False,
            )

            pregrasp, _, top_source = self.pregrasp_realign_pose(grasp, move_quat)
            ok = self.run_task(
                f"{task_id}_pregrasp_realign",
                f"{task_desc}: pregrasp realign",
                [make_move_step("move_to_pregrasp_realign_height", pregrasp, self.args.descend_speed, move_quat)],
            )
            if not ok:
                raise PickExecutionError(
                    "Pick task failed during pregrasp realign move", phase="pregrasp", retryable=True
                )
            original_pregrasp = pregrasp
            pregrasp = self.realign_contact("pregrasp", pregrasp, self.args.descend_speed, move_quat)
            grasp, lift = self.apply_realign_delta_to_descent(original_pregrasp, pregrasp, grasp, lift)
            self.update_selected_execution_pose(
                pregrasp=pregrasp,
                grasp=grasp,
                lift=lift,
                quat_xyzw=move_quat,
                pregrasp_top_source=top_source,
            )
            self.sample_pick_diagnostics(
                "pregrasp",
                pregrasp,
                commanded_quat_xyzw=move_quat,
                detect_target=False,
            )

            ok = self.run_task(
                f"{task_id}_grasp",
                f"{task_desc}: grasp",
                [make_move_step("descend_to_graspgen_pose_no_realign", grasp, self.args.descend_speed, move_quat)],
            )
            if not ok:
                raise PickExecutionError("Pick task failed during grasp", phase="grasp", retryable=True)
            self.log_grasp_contact_residual(grasp, move_quat)
            self.sample_pick_diagnostics("grasp", grasp, commanded_quat_xyzw=move_quat, detect_target=True)

            ok = self.run_task(
                f"{task_id}_close",
                f"{task_desc}: close",
                [
                    make_gripper_step("close_gripper_on_target", 0.0),
                    make_wait_step("hold_target", self.args.hold_s),
                ],
            )
            if not ok:
                raise PickExecutionError("Pick task failed during close", phase="close", retryable=False)
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
                raise PickExecutionError("Pick task failed during lift", phase="lift", retryable=False)
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

    def execute_graspgen_candidates(self, candidates: list[dict[str, object]]) -> None:
        last_error: Exception | None = None
        max_attempts = int(self.args.max_execution_attempts)
        attempt_candidates = candidates if max_attempts <= 0 else candidates[: max(1, max_attempts)]
        print(
            f"GRASPGEN_EXECUTION_CANDIDATES total={len(candidates)} attempts={len(attempt_candidates)} "
            f"retry_after_grasp_residual={self.args.retry_after_grasp_residual}",
            flush=True,
        )
        for attempt, candidate in enumerate(attempt_candidates, start=1):
            index = int(candidate["index"])
            self.current_execution_candidate_index = index
            self.selected_target_contact_ee = tuple(float(v) for v in candidate["target_contact"])
            self.selected_plan_contact_base = tuple(float(v) for v in candidate["plan_contact"])
            self.realign_target_contact_base_by_phase = {}
            self.mark_execution_candidate_attempt(
                index,
                selected=True,
                stage="selected",
                reason=f"execution_attempt_{attempt}",
            )
            self._write_execution_debug_outputs()
            print(
                f"GRASPGEN_EXECUTION_ATTEMPT attempt={attempt} idx={index} "
                f"approach={fmt_xyz(candidate['approach'])} grasp={fmt_xyz(candidate['grasp'])}",
                flush=True,
            )
            try:
                self.execute_pick(
                    candidate["approach"],
                    candidate["grasp"],
                    candidate["lift"],
                    candidate["quat"],
                )
                self.mark_execution_candidate_attempt(
                    index,
                    selected=True,
                    stage="selected",
                    reason=f"executed_successfully_attempt_{attempt}",
                )
                return
            except PickExecutionError as exc:
                last_error = exc
                self.mark_execution_candidate_failed(index, exc.phase, str(exc))
                self._write_execution_debug_outputs()
                residual_retry_disabled = exc.phase == "grasp_residual" and not self.args.retry_after_grasp_residual
                has_next_candidate = attempt < len(attempt_candidates)
                will_retry = bool(exc.retryable and not residual_retry_disabled and has_next_candidate)
                event = "GRASPGEN_EXECUTION_RETRY" if will_retry else "GRASPGEN_EXECUTION_STOP"
                print(
                    f"{event} attempt={attempt} idx={index} phase={exc.phase} retryable={exc.retryable} "
                    f"will_retry={will_retry} error={exc}",
                    flush=True,
                )
                if residual_retry_disabled:
                    raise
                if not exc.retryable:
                    raise
                continue

        raise RuntimeError(
            f"All attempted GraspGen execution candidates failed "
            f"({len(attempt_candidates)}/{len(candidates)} tried); last_error={last_error}"
        )


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
            candidates = node.select_graspgen_candidates(base_to_gripper_tf)
            node.execute_graspgen_candidates(candidates)
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
        node._write_execution_debug_outputs()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
