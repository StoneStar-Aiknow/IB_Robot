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
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState

from ibrobot_msgs.action import ExecuteTaskPlan
from ibrobot_msgs.msg import Detection2D, TaskStep
from ibrobot_msgs.srv import DetectSegment, MoveToConfiguration, PlanGrasp, VerifyGrasp
from manipulation_execution.contact_compensation import ContactPrediction, compensate_contact_xy
from manipulation_execution.grasp_geometry import (
    FixedFingerRobustGap,
    canonicalize_joint5,
    fixed_finger_base_side_alignment,
    fixed_finger_envelope_score,
    fixed_finger_robust_gap,
    grasp_axis_errors,
    joint5_closing_axis_correction,
    prepared_candidate_soft_score,
    target_width_extent_points,
)

JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG = 20.0
SO101_GEOMETRY_THRESHOLD_FALLBACK_M = 1e-5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        default=False,
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
        help=(
            "Logging threshold for low-height contact XY residual. "
            "No retract or low-height realign is executed; <=0 disables this log annotation."
        ),
    )
    parser.add_argument(
        "--grasp-residual-abort-xy-error",
        type=float,
        default=0.030,
        help=(
            "Logging threshold for large low-height contact XY residual. "
            "No abort or retract is executed; <=0 disables this log annotation."
        ),
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
        default=True,
        help="After a safe pregrasp retreat, retry the next candidate when the fixed-finger robust gap is insufficient.",
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
        default=False,
        help="Re-run target detection at the grasp pose; disabled by default because close-range masks are unreliable",
    )
    parser.add_argument(
        "--pick-diagnostics-settle-s",
        type=float,
        default=0.25,
        help="Settling time before sampling pick diagnostic TF/detection",
    )
    parser.add_argument(
        "--pick-diagnostics-max-target-contact-distance",
        type=float,
        default=0.08,
        help="Do not replace the trusted target when a close-range diagnostic detection is farther from contact",
    )
    parser.add_argument(
        "--grasp-verification",
        choices=("required", "optional", "disabled"),
        default="required",
        help="Post-close/lift verification policy; optional skips only when the verifier service is unavailable",
    )
    parser.add_argument(
        "--completion-mode",
        choices=("retained", "pick"),
        default="retained",
        help=(
            "Completion policy: retained releases immediately after close/probe retention verification; "
            "pick continues through final lift and optional post-success release"
        ),
    )
    parser.add_argument(
        "--grasp-verification-service",
        default="/grasp_verifier/verify_grasp",
        help="VerifyGrasp service used to confirm object retention",
    )
    parser.add_argument(
        "--grasp-verification-timeout-s",
        type=float,
        default=5.0,
        help="Timeout for each post-grasp verification call",
    )
    parser.add_argument(
        "--grasp-verification-wait-s",
        type=float,
        default=0.1,
        help="Additional verifier-side settling delay before each sensor sample",
    )
    parser.add_argument(
        "--grasp-verification-probe-lift-height",
        type=float,
        default=0.03,
        help="Height of the slow retention-check lift before the final lift; <=0 disables the probe",
    )
    parser.add_argument(
        "--grasp-verification-probe-lift-speed",
        type=float,
        default=0.10,
        help="Velocity scaling for the retention-check probe lift",
    )
    parser.add_argument(
        "--recover-after-close-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retract closed to pregrasp, open, and return to the observation pose after close verification fails",
    )
    parser.add_argument(
        "--recover-after-retention-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open at the elevated pose and return to observation after probe or final retention verification fails",
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
        help="Maximum candidates sent to IK after inexpensive geometry filters; <=0 tests all eligible candidates",
    )
    parser.add_argument(
        "--prepared-candidate-scoring",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override robot_config and enable or disable IK/FK-prepared candidate soft ranking",
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
    parser.add_argument("--fk-service", default="/compute_fk", help="MoveIt GetPositionFK service name")
    parser.add_argument(
        "--final-joint5-max",
        type=float,
        default=2.0,
        help="Reject the 180-degree-adjusted grasp IK when |joint5| exceeds this safety limit",
    )
    parser.add_argument(
        "--ik-worker-count",
        type=int,
        default=0,
        help="Number of isolated MoveIt IK/FK workers used for candidate preparation; 0 keeps the serial path",
    )
    parser.add_argument(
        "--ik-worker-prefix",
        default="/ik_worker",
        help="Worker namespace prefix; worker services are <prefix>_<index>/compute_ik and compute_fk",
    )
    parser.add_argument(
        "--move-configuration-service",
        default="/moveit_gateway/move_to_configuration",
        help="MoveIt gateway service used to execute the exact IK solution",
    )
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
        "--ik-fk-contact-compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use IK then FK to iteratively compensate final grasp contact error along base-frame X and Y",
    )
    parser.add_argument(
        "--ik-fk-contact-tolerance",
        type=float,
        default=0.003,
        help="Maximum predicted base-frame X/Y contact residual after IK/FK compensation",
    )
    parser.add_argument(
        "--ik-fk-contact-max-iterations",
        type=int,
        default=3,
        help="Maximum base-frame X/Y correction updates after the initial IK/FK prediction",
    )
    parser.add_argument(
        "--ik-fk-contact-max-correction",
        type=float,
        default=0.030,
        help="Maximum absolute base-frame X/Y command correction allowed for one grasp",
    )
    parser.add_argument(
        "--ik-fk-contact-max-xz-error",
        type=float,
        default=0.020,
        help="Maximum IK/FK-predicted contact error along the uncorrected base-frame Z axis",
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

    parser.add_argument("--observe-speed", type=float, default=0.20, help="Velocity scaling for observation move")
    parser.add_argument("--approach-speed", type=float, default=0.25, help="Velocity scaling for approach move")
    parser.add_argument("--descend-speed", type=float, default=0.15, help="Velocity scaling for descend move")
    parser.add_argument("--lift-speed", type=float, default=0.25, help="Velocity scaling for lift move")
    parser.add_argument("--observe-settle-s", type=float, default=0.6, help="Wait after reaching observation pose")
    parser.add_argument("--open-settle-s", type=float, default=0.3, help="Wait after opening gripper")
    parser.add_argument("--hold-s", type=float, default=0.8, help="Wait after closing gripper before lift")
    parser.add_argument(
        "--release-after-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the gripper at the final lift pose after successful retention verification",
    )
    parser.add_argument(
        "--release-settle-s",
        type=float,
        default=1.0,
        help="Wait after a requested post-success release so the target can fall clear",
    )
    parser.add_argument(
        "--release-drop-height-m",
        type=float,
        default=-1.0,
        help="Descend to this height above the grasp pose before release; negative keeps the final lift pose",
    )

    parser.add_argument(
        "--detect-service", default="/grounded_sam2/detect_and_segment", help="DetectSegment service name"
    )
    parser.add_argument("--task-action", default="/task_executor/execute_task_plan", help="ExecuteTaskPlan action name")
    parser.add_argument("--ready-timeout-s", type=float, default=12.0, help="Service/action wait timeout")
    parser.add_argument("--detect-timeout-s", type=float, default=90.0, help="Detection call timeout")
    parser.add_argument("--task-timeout-s", type=float, default=150.0, help="Task execution timeout")
    parser.add_argument("--task-goal-timeout-s", type=float, default=30.0, help="Task action goal acceptance timeout")
    parser.add_argument("--tf-timeout-s", type=float, default=10.0, help="TF lookup timeout")

    cli_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(cli_args)
    args._explicit_cli_destinations = {  # noqa: SLF001 - script-local configuration state.
        action.dest
        for action in parser._actions
        if any(
            token == option or token.startswith(f"{option}=") for token in cli_args for option in action.option_strings
        )
    }
    args._graspgen_to_ee_translation_auto = (  # noqa: SLF001 - script-local diagnostic state.
        args.graspgen_to_ee_x is None and args.graspgen_to_ee_y is None and args.graspgen_to_ee_z is None
    )
    load_grasp_execution_config(args)
    if args.final_joint5_max is not None and (not math.isfinite(args.final_joint5_max) or args.final_joint5_max <= 0.0):
        raise ValueError("--final-joint5-max must be finite and positive")
    if args.prepared_candidate_scoring is not None:
        args.prepared_candidate_scoring_config["enabled"] = bool(args.prepared_candidate_scoring)
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
    args.target_gripper_type = ""
    args.target_fixed_finger_contact_ee = None
    args.target_closing_axis_ee = None
    args.target_fixed_finger_margin_m = 0.0
    args.target_fixed_finger_margin_max_m = 0.0
    args.target_fixed_finger_margin_width_ref_m = 0.035
    args.target_fixed_finger_margin_width_gain = 0.0
    args.target_fixed_finger_base_side_config = {
        "enabled": False,
        "reference_point_base": (0.0, 0.0, 0.0),
        "min_alignment_cos": 0.0,
    }
    args.target_fixed_finger_robust_gap_config = {
        "enabled": False,
        "max_target_gap_deficit_m": 0.003,
    }
    args.target_ik_orientation_guard_config = {
        "enabled": False,
        "approach_axis_ee": (0.0, 0.0, 1.0),
        "closing_axis_180_symmetric": False,
        "max_approach_error_deg": 25.0,
        "max_closing_error_deg": 20.0,
    }
    args.target_width_clearance_m = 0.003
    args.target_width_min_m = 0.005
    args.target_width_max_m = 0.08
    args.target_width_fallback_m = 0.035
    args.execution_scoring_config = {}
    args.execution_contact_distance_weight = 1.0
    args.execution_topdown_weight = float(args.graspgen_topdown_weight)
    args.execution_confidence_weight = 1.0
    args.execution_contact_distance_scale = float(args.graspgen_contact_distance_scale)
    args.prepared_candidate_scoring_config = {
        "enabled": False,
        "fixed_finger_envelope_weight": 0.55,
        "contact_xy_weight": 0.25,
        "contact_z_weight": 0.15,
        "confidence_weight": 0.05,
        "centroid_distance_weight": 0.50,
        "contact_xy_scale_m": 0.030,
        "contact_z_scale_m": 0.020,
        "centroid_distance_scale_m": 0.010,
        "fixed_finger_gap_sigma_m": 0.006,
        "missing_fixed_finger_envelope_score": 0.20,
        "fixed_finger_score_weight": 0.80,
        "reliable_max_opening_m": 0.072,
        "moving_finger_min_clearance_m": 0.003,
    }

    explicit_cli_destinations = getattr(args, "_explicit_cli_destinations", set())

    def apply_config_value(destination: str, value, *, cli_destination: str | None = None) -> None:
        if (cli_destination or destination) not in explicit_cli_destinations:
            setattr(args, destination, value)

    if args.robot_config is None or not args.robot_config.exists():
        return

    payload = yaml.safe_load(args.robot_config.read_text(encoding="utf-8")) or {}
    robot = payload.get("robot") if isinstance(payload, dict) else None
    if not isinstance(robot, dict):
        return

    config = robot.get("grasp_execution")
    if not isinstance(config, dict):
        return

    direct_options = (
        ("planner_service", "manipulation_service", str),
        ("verifier_service", "grasp_verification_service", str),
        ("detect_service", "detect_service", str),
        ("ik_service", "ik_service", str),
        ("fk_service", "fk_service", str),
        ("move_configuration_service", "move_configuration_service", str),
        ("base_frame", "base_frame", str),
        ("ee_frame", "ee_frame", str),
        ("timeout_sec", "task_timeout_s", float),
        ("ready_timeout_sec", "ready_timeout_s", float),
        ("verification", "grasp_verification", str),
        ("verification_timeout_sec", "grasp_verification_timeout_s", float),
        ("verification_wait_sec", "grasp_verification_wait_s", float),
        ("recover_after_close_failure", "recover_after_close_failure", bool),
        ("recover_after_retention_failure", "recover_after_retention_failure", bool),
        ("max_execution_attempts", "max_execution_attempts", int),
        ("approach_distance_m", "graspgen_approach_distance", float),
        ("lift_distance_m", "final_lift", float),
        ("observe_velocity_scaling", "observe_speed", float),
        ("approach_velocity_scaling", "approach_speed", float),
        ("descend_velocity_scaling", "descend_speed", float),
        ("probe_lift_velocity_scaling", "grasp_verification_probe_lift_speed", float),
        ("lift_velocity_scaling", "lift_speed", float),
        ("observe_settle_sec", "observe_settle_s", float),
        ("open_settle_sec", "open_settle_s", float),
        ("hold_sec", "hold_s", float),
        ("probe_lift_height_m", "grasp_verification_probe_lift_height", float),
    )
    for config_key, destination, cast in direct_options:
        if config_key in config:
            apply_config_value(destination, cast(config[config_key]))

    planner = config.get("planner", {})
    if isinstance(planner, dict):
        planner_options = (
            ("confidence_threshold", "confidence_threshold", float),
            ("grasp_threshold", "grasp_threshold", float),
            ("timeout_sec", "detect_timeout_s", float),
            ("debug_output_mode", "debug_output_mode", str),
        )
        for config_key, destination, cast in planner_options:
            if config_key in planner:
                apply_config_value(destination, cast(planner[config_key]))

    candidate_selection = config.get("candidate_selection", {})
    if isinstance(candidate_selection, dict):
        candidate_options = (
            ("min_confidence", "min_grasp_confidence", float),
            ("min_point_count", "min_point_count", int),
            ("require_collision_free", "require_collision_free_grasp", bool),
            ("min_contact_z", "min_contact_z", float),
            ("min_approach_z", "min_approach_z", float),
            ("topdown_min_z", "graspgen_topdown_min_z", float),
            ("max_candidates", "max_candidates", int),
        )
        for config_key, destination, cast in candidate_options:
            if config_key in candidate_selection:
                apply_config_value(destination, cast(candidate_selection[config_key]))
        if "confidence_weight" in candidate_selection:
            args.execution_confidence_weight = float(candidate_selection["confidence_weight"])
        if "topdown_weight" in candidate_selection:
            apply_config_value(
                "execution_topdown_weight",
                float(candidate_selection["topdown_weight"]),
                cli_destination="graspgen_topdown_weight",
            )

    ik = config.get("ik", {})
    if isinstance(ik, dict):
        ik_options = (
            ("group_name", "ik_group", str),
            ("timeout_sec", "ik_timeout_s", float),
            ("avoid_collisions", "ik_avoid_collisions", bool),
            ("check_orientation", "ik_check_orientation", bool),
            ("worker_count", "ik_worker_count", int),
            ("worker_namespace_prefix", "ik_worker_prefix", str),
        )
        for config_key, destination, cast in ik_options:
            if config_key in ik:
                apply_config_value(destination, cast(ik[config_key]))

    contact_compensation = config.get("contact_compensation", {})
    if isinstance(contact_compensation, dict):
        compensation_options = (
            ("enabled", "ik_fk_contact_compensation", bool),
            ("xy_tolerance_m", "ik_fk_contact_tolerance", float),
            ("max_iterations", "ik_fk_contact_max_iterations", int),
            ("max_correction_m", "ik_fk_contact_max_correction", float),
            ("max_z_error_m", "ik_fk_contact_max_xz_error", float),
        )
        for config_key, destination, cast in compensation_options:
            if config_key in contact_compensation:
                apply_config_value(destination, cast(contact_compensation[config_key]))

    contact_realign = config.get("contact_realign", {})
    if isinstance(contact_realign, dict):
        realign_options = (
            ("enabled", "contact_realign", bool),
            ("tolerance_m", "contact_realign_tolerance", float),
            ("max_iterations", "contact_realign_max_iterations", int),
            ("pregrasp_clearance_m", "pregrasp_realign_clearance", float),
        )
        for config_key, destination, cast in realign_options:
            if config_key in contact_realign:
                apply_config_value(destination, cast(contact_realign[config_key]))

    pose_diagnostics = config.get("pose_diagnostics", {})
    if isinstance(pose_diagnostics, dict):
        diagnostic_options = (
            ("enabled", "pick_diagnostics", bool),
            ("settle_sec", "pick_diagnostics_settle_s", float),
            ("grasp_warn_threshold_m", "grasp_realign_max_xy_error", float),
            ("grasp_realign_log_threshold_m", "grasp_residual_realign_xy_error", float),
            ("grasp_abort_log_threshold_m", "grasp_residual_abort_xy_error", float),
        )
        for config_key, destination, cast in diagnostic_options:
            if config_key in pose_diagnostics:
                apply_config_value(destination, cast(pose_diagnostics[config_key]))

    target_geometry = config.get("target_geometry", {})
    if isinstance(target_geometry, dict):
        geometry_options = (
            ("tabletop_filter", "so101_tabletop_filter", bool),
            ("tabletop_clearance_m", "so101_tabletop_clearance", float),
            ("tabletop_sweep_steps", "so101_tabletop_sweep_steps", int),
        )
        for config_key, destination, cast in geometry_options:
            if config_key in target_geometry:
                apply_config_value(destination, cast(target_geometry[config_key]))

    target_gripper = config.get("target_gripper", {})
    if not isinstance(target_gripper, dict):
        target_gripper = {}
    args.target_gripper_type = str(target_gripper.get("type", ""))

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
    orientation_guard = target_gripper.get("ik_orientation_guard", {})
    if isinstance(orientation_guard, dict):
        args.target_ik_orientation_guard_config.update(orientation_guard)
        if "joint5_abs_max" in orientation_guard:
            apply_config_value("final_joint5_max", float(orientation_guard["joint5_abs_max"]))
    args.target_ik_orientation_guard_config["approach_axis_ee"] = tuple(
        float(v)
        for v in _normalized_vector(
            args.target_ik_orientation_guard_config["approach_axis_ee"],
            "grasp_execution.target_gripper.ik_orientation_guard.approach_axis_ee",
        )
    )
    for key in ("max_approach_error_deg", "max_closing_error_deg"):
        value = float(args.target_ik_orientation_guard_config[key])
        if not math.isfinite(value) or not 0.0 <= value <= 180.0:
            raise ValueError(f"grasp_execution.target_gripper.ik_orientation_guard.{key} must be in [0, 180]")
        args.target_ik_orientation_guard_config[key] = value
    if bool(args.target_ik_orientation_guard_config["enabled"]) and args.target_closing_axis_ee is None:
        raise ValueError(
            "grasp_execution.target_gripper.closing_axis_ee is required when ik_orientation_guard is enabled"
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
    base_side = target_gripper.get("fixed_finger_base_side", {})
    if isinstance(base_side, dict):
        args.target_fixed_finger_base_side_config.update(base_side)
    args.target_fixed_finger_base_side_config["reference_point_base"] = _float_triplet(
        args.target_fixed_finger_base_side_config["reference_point_base"],
        "grasp_execution.target_gripper.fixed_finger_base_side.reference_point_base",
    )
    min_alignment_cos = float(args.target_fixed_finger_base_side_config["min_alignment_cos"])
    if not -1.0 <= min_alignment_cos <= 1.0:
        raise ValueError("grasp_execution.target_gripper.fixed_finger_base_side.min_alignment_cos must be in [-1, 1]")
    args.target_fixed_finger_base_side_config["min_alignment_cos"] = min_alignment_cos
    robust_gap = target_gripper.get("fixed_finger_robust_gap", {})
    if isinstance(robust_gap, dict):
        args.target_fixed_finger_robust_gap_config.update(robust_gap)
    max_target_gap_deficit = float(args.target_fixed_finger_robust_gap_config["max_target_gap_deficit_m"])
    if max_target_gap_deficit < 0.0:
        raise ValueError(
            "grasp_execution.target_gripper.fixed_finger_robust_gap.max_target_gap_deficit_m must be non-negative"
        )
    args.target_fixed_finger_robust_gap_config["max_target_gap_deficit_m"] = max_target_gap_deficit
    if "width_clearance_m" in target_gripper:
        args.target_width_clearance_m = float(target_gripper["width_clearance_m"])
    if "min_width_m" in target_gripper:
        args.target_width_min_m = float(target_gripper["min_width_m"])
    if "max_width_m" in target_gripper:
        args.target_width_max_m = float(target_gripper["max_width_m"])
    if "fallback_width_m" in target_gripper:
        args.target_width_fallback_m = float(target_gripper["fallback_width_m"])
    if "width_quality_min" in target_gripper:
        apply_config_value("target_width_quality_min", float(target_gripper["width_quality_min"]))

    scoring = config.get("execution_scoring", {})
    if isinstance(scoring, dict):
        args.execution_scoring_config = scoring
        if "contact_distance_weight" in scoring:
            args.execution_contact_distance_weight = float(scoring["contact_distance_weight"])
        if "topdown_weight" in scoring:
            apply_config_value(
                "execution_topdown_weight",
                float(scoring["topdown_weight"]),
                cli_destination="graspgen_topdown_weight",
            )
        if "confidence_weight" in scoring:
            args.execution_confidence_weight = float(scoring["confidence_weight"])
        if "contact_distance_scale_m" in scoring:
            apply_config_value(
                "execution_contact_distance_scale",
                float(scoring["contact_distance_scale_m"]),
                cli_destination="graspgen_contact_distance_scale",
            )

    prepared_scoring = config.get("prepared_candidate_scoring", {})
    if isinstance(prepared_scoring, dict):
        args.prepared_candidate_scoring_config.update(prepared_scoring)

    source_contact = config.get("source_contact_point")
    if source_contact is not None:
        source_x, source_y, source_z = _float_triplet(source_contact, "grasp_execution.source_contact_point")
        apply_config_value("graspgen_contact_x", source_x)
        apply_config_value("graspgen_contact_y", source_y)
        apply_config_value("graspgen_contact_z", source_z)

    adapter = config.get("adapter", {})
    if isinstance(adapter, dict):
        rpy = adapter.get("source_to_ee_rpy")
        if rpy is not None:
            roll, pitch, yaw = _float_triplet(rpy, "grasp_execution.adapter.source_to_ee_rpy")
            apply_config_value("graspgen_to_ee_roll", roll)
            apply_config_value("graspgen_to_ee_pitch", pitch)
            apply_config_value("graspgen_to_ee_yaw", yaw)

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


def resolve_target_contact_for_candidate(
    args: argparse.Namespace,
    candidate,
) -> tuple[np.ndarray, str, float | None]:
    static_contact = _static_target_contact(args)
    if not args.target_auto_width_compensation:
        return static_contact, "static:auto_disabled", None
    if not args._graspgen_to_ee_translation_auto:
        return static_contact, "static:manual_adapter", None
    if args.target_fixed_finger_contact_ee is None or args.target_closing_axis_ee is None:
        return static_contact, "static:missing_target_gripper_config", None

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
    fixed_finger_target_gap = center_offset - 0.5 * width
    return contact, reason, fixed_finger_target_gap


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


@dataclass(frozen=True)
class IKFKContactPayload:
    joint_state: JointState
    ee_xyz: tuple[float, float, float]
    ee_quat_xyzw: tuple[float, float, float, float]
    joint5_retry_applied: bool = False
    original_joint5: float | None = None
    approach_axis_error_deg: float | None = None
    closing_axis_error_deg: float | None = None


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
        self.use_grasp_verification = args.grasp_verification != "disabled"
        self.verify_grasp_client = (
            self.create_client(VerifyGrasp, args.grasp_verification_service) if self.use_grasp_verification else None
        )
        self.use_ik_fk_contact_compensation = bool(args.ik_fk_contact_compensation and args.target_source == "graspgen")
        orientation_guard_enabled = bool(args.target_ik_orientation_guard_config.get("enabled", False))
        needs_ik_services = (
            args.ik_filter
            or self.use_ik_fk_contact_compensation
            or args.final_joint5_max is not None
            or orientation_guard_enabled
        )
        needs_fk_services = (
            self.use_ik_fk_contact_compensation or args.final_joint5_max is not None or orientation_guard_enabled
        )
        self.ik_client = self.create_client(GetPositionIK, args.ik_service) if needs_ik_services else None
        self.fk_client = self.create_client(GetPositionFK, args.fk_service) if needs_fk_services else None
        worker_count = max(0, int(args.ik_worker_count))
        worker_prefix = str(args.ik_worker_prefix).rstrip("/")
        if worker_count and not worker_prefix:
            raise ValueError("--ik-worker-prefix must not be empty when --ik-worker-count is positive")
        self.ik_worker_clients = []
        self.fk_worker_clients = []
        self._ik_worker_callback_group = ReentrantCallbackGroup()
        self._parallel_ik_seed: JointState | None = None
        self._parallel_ik_seed_subscription = None
        if needs_ik_services:
            for index in range(worker_count):
                namespace = f"{worker_prefix}_{index}"
                self.ik_worker_clients.append(
                    self.create_client(
                        GetPositionIK,
                        f"{namespace}/compute_ik",
                        callback_group=self._ik_worker_callback_group,
                    )
                )
                if needs_fk_services:
                    self.fk_worker_clients.append(
                        self.create_client(
                            GetPositionFK,
                            f"{namespace}/compute_fk",
                            callback_group=self._ik_worker_callback_group,
                        )
                    )
        if worker_count:
            self._parallel_ik_seed_subscription = self.create_subscription(
                JointState,
                "/joint_states",
                self._parallel_ik_seed_callback,
                10,
                callback_group=self._ik_worker_callback_group,
            )
        self.move_configuration_client = (
            self.create_client(MoveToConfiguration, args.move_configuration_service) if needs_fk_services else None
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.handeye_data, self.handeye_matrix = self._load_handeye(args)
        self.selected_target_contact_ee: tuple[float, float, float] | None = None
        self.selected_plan_contact_base: tuple[float, float, float] | None = None
        self.selected_target_width_m = 0.0
        self.selected_fixed_finger_envelope = None
        self.selected_target_width_extent_base = None
        self.selected_grasp_joint_seed: JointState | None = None
        self.active_grasp_joint_seed: JointState | None = None
        self.current_grasp_verified = False
        self.current_execution_candidate_index: int | None = None
        self.observed_target_base: tuple[float, float, float] | None = None
        self.observed_target_base_alt: tuple[float, float, float] | None = None
        self.current_plan_contact_base: tuple[float, float, float] | None = None
        self.realign_target_contact_base_by_phase: dict[str, tuple[float, float, float]] = {}
        self.last_graspgen_debug_output_dir: Path | None = None
        self.last_graspgen_surface_centroid_camera: tuple[float, float, float] | None = None
        self.last_graspgen_volume_centroid_camera: tuple[float, float, float] | None = None
        self.last_graspgen_object_top_camera: tuple[float, float, float] | None = None
        self.execution_debug_records: list[dict[str, object]] = []
        self.prepared_candidate_ranking_records: list[dict[str, object]] = []
        self.pick_diagnostic_records: list[dict[str, object]] = []
        self.grasp_verification_records: list[dict[str, object]] = []
        self.last_t_base_camera: np.ndarray | None = None
        self._so101_table_plane: tuple[np.ndarray, float, float] | None = None
        self._so101_table_plane_shadow: tuple[np.ndarray, float, float] | None = None
        self._so101_table_plane_checked = False
        self._so101_table_shadow_comparisons = 0
        self._so101_table_shadow_mismatches = 0
        self._so101_table_shadow_max_clearance_delta = 0.0

    def _parallel_ik_seed_callback(self, message: JointState) -> None:
        self._parallel_ik_seed = message

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
        needs_verifier = not self.args.detect_only and not self.args.observe_only and self.use_grasp_verification
        if needs_verifier and self.verify_grasp_client is not None:
            verifier_ready = self.verify_grasp_client.wait_for_service(timeout_sec=self.args.ready_timeout_s)
            if not verifier_ready and self.args.grasp_verification == "required":
                raise RuntimeError(
                    f"Grasp verification service is not available: {self.args.grasp_verification_service}"
                )
            if not verifier_ready:
                print(
                    f"GRASP_VERIFY_READY available=False policy={self.args.grasp_verification} "
                    f"service={self.args.grasp_verification_service}",
                    flush=True,
                )
        if (
            not self.args.detect_only
            and not self.args.observe_only
            and self.use_ik_fk_contact_compensation
            and self.move_configuration_client is not None
            and not self.move_configuration_client.wait_for_service(timeout_sec=self.args.ready_timeout_s)
        ):
            raise RuntimeError(f"Move configuration service is not available: {self.args.move_configuration_service}")

    def wait_ik_ready(self) -> None:
        orientation_guard_enabled = bool(self.args.target_ik_orientation_guard_config.get("enabled", False))
        if (
            (
                self.args.ik_filter
                or self.use_ik_fk_contact_compensation
                or self.args.final_joint5_max is not None
                or orientation_guard_enabled
            )
            and self.ik_client is not None
            and not self.ik_client.wait_for_service(timeout_sec=self.args.ik_wait_timeout_s)
        ):
            raise RuntimeError(f"IK service is not available: {self.args.ik_service}")
        if self.fk_client is not None and not self.fk_client.wait_for_service(timeout_sec=self.args.ik_wait_timeout_s):
            raise RuntimeError(f"FK service is not available: {self.args.fk_service}")
        worker_prefix = str(self.args.ik_worker_prefix).rstrip("/")
        for index, client in enumerate(self.ik_worker_clients):
            service = f"{worker_prefix}_{index}/compute_ik"
            if not client.wait_for_service(timeout_sec=self.args.ik_wait_timeout_s):
                raise RuntimeError(f"Parallel IK worker service is not available: {service}")
        for index, client in enumerate(self.fk_worker_clients):
            service = f"{worker_prefix}_{index}/compute_fk"
            if not client.wait_for_service(timeout_sec=self.args.ik_wait_timeout_s):
                raise RuntimeError(f"Parallel FK worker service is not available: {service}")
        if self.ik_worker_clients:
            deadline = time.monotonic() + float(self.args.ik_wait_timeout_s)
            while self._parallel_ik_seed is None and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
            if self._parallel_ik_seed is None:
                raise RuntimeError("Parallel IK worker pool requires a current /joint_states sample")
            print(
                f"IK_WORKER_POOL ready=True workers={len(self.ik_worker_clients)} prefix={worker_prefix} "
                f"seed_joints={len(self._parallel_ik_seed.name)}",
                flush=True,
            )

    def run_task(self, task_id: str, description: str, steps: list[TaskStep], timeout_s: float | None = None) -> bool:
        started = time.monotonic()
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
            f"duration={result.total_duration_s:.2f}s wall_s={time.monotonic() - started:.3f} msg={result.message}",
            flush=True,
        )
        return bool(result.success)

    def run_joint_configuration(self, label: str, joint_state: JointState, velocity_scaling: float) -> bool:
        started = time.monotonic()
        client = self.move_configuration_client
        if client is None:
            raise RuntimeError("Move configuration client is disabled")
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=self.args.ready_timeout_s):
            raise RuntimeError(f"Move configuration service is not available: {self.args.move_configuration_service}")

        request = MoveToConfiguration.Request()
        request.target_joint_state = joint_state
        request.velocity_scaling = float(velocity_scaling)
        print(
            f"MOVE_CONFIGURATION_SEND label={label} joints={list(joint_state.name)} "
            f"positions={[round(float(value), 6) for value in joint_state.position]}",
            flush=True,
        )
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.task_timeout_s)
        if not future.done():
            raise RuntimeError(f"Timed out waiting for move configuration: {label}")
        response = future.result()
        if response is None:
            raise RuntimeError(f"Move configuration returned no response: {label}")
        print(
            f"MOVE_CONFIGURATION_RESULT label={label} success={response.success} "
            f"duration={response.execution_time_s:.2f}s wall_s={time.monotonic() - started:.3f} msg={response.message}",
            flush=True,
        )
        return bool(response.success)

    def run_branch_locked_pose(
        self,
        label: str,
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
        velocity_scaling: float,
        *,
        phase: str,
        retryable: bool = True,
        validate_orientation: bool = True,
    ) -> IKFKContactPayload:
        seed = self.active_grasp_joint_seed
        payload, code, failed_reason = self.solve_grasp_ik_fk(
            label,
            xyz,
            quat_xyzw,
            seed,
            validate_orientation=validate_orientation,
        )
        if payload is None:
            raise PickExecutionError(
                f"Branch-locked IK failed: code={code} reason={failed_reason}",
                phase=phase,
                retryable=retryable,
            )
        if not validate_orientation:
            approach_error = payload.approach_axis_error_deg
            closing_error = payload.closing_axis_error_deg
            approach_text = "n/a" if approach_error is None else f"{approach_error:.3f}"
            closing_text = "n/a" if closing_error is None else f"{closing_error:.3f}"
            print(
                f"IK_ORIENTATION_GATE label={label} enabled=False "
                f"approach_error_deg={approach_text} closing_error_deg={closing_text}",
                flush=True,
            )
        try:
            self._validate_joint5_branch_continuity(label, seed, payload.joint_state)
        except RuntimeError as exc:
            raise PickExecutionError(str(exc), phase=phase, retryable=retryable) from exc
        if not self.run_joint_configuration(label, payload.joint_state, velocity_scaling):
            raise PickExecutionError(
                f"Branch-locked joint motion failed: {label}",
                phase=phase,
                retryable=retryable,
            )
        self.active_grasp_joint_seed = payload.joint_state
        return payload

    def verify_grasp_retention(self, label: str) -> None:
        if not self.use_grasp_verification:
            return

        client = self.verify_grasp_client
        if client is None or not client.service_is_ready():
            if self.args.grasp_verification == "optional":
                print(
                    f"GRASP_VERIFY label={label} skipped=True reason=service_unavailable "
                    f"service={self.args.grasp_verification_service}",
                    flush=True,
                )
                return
            raise PickExecutionError(
                f"Grasp verification service is unavailable: {self.args.grasp_verification_service}",
                phase=f"verify_{label}",
                retryable=False,
            )

        started = time.monotonic()
        request = VerifyGrasp.Request()
        request.task_id = f"banana_pick_{self.current_execution_candidate_index}_{label}"
        request.text_prompt = self.args.prompt
        request.expected_target_width_m = float(self.selected_target_width_m)
        request.post_grasp_wait_s = max(0.0, float(self.args.grasp_verification_wait_s))

        future = client.call_async(request)
        timeout_s = max(0.1, float(self.args.grasp_verification_timeout_s)) + request.post_grasp_wait_s
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            if self.args.grasp_verification == "optional":
                print(f"GRASP_VERIFY label={label} skipped=True reason=timeout", flush=True)
                return
            raise PickExecutionError(
                f"Grasp verification timed out after {timeout_s:.1f}s",
                phase=f"verify_{label}",
                retryable=False,
            )

        response = future.result()
        status = int(response.status)
        status_name = {0: "failed", 1: "success", 2: "uncertain"}.get(status, f"unknown_{status}")
        record = {
            "label": label,
            "success": bool(response.success),
            "status": status,
            "status_name": status_name,
            "confidence": round(float(response.confidence), 6),
            "message": response.message,
            "expected_target_width_m": round(float(self.selected_target_width_m), 6),
            "evidence": list(response.evidence),
        }
        if self.current_execution_candidate_index is not None:
            record["candidate_index"] = int(self.current_execution_candidate_index)
        self.grasp_verification_records.append(record)
        selected_record = (
            self._execution_record_for_index(self.current_execution_candidate_index)
            if self.current_execution_candidate_index is not None
            else None
        )
        if selected_record is not None:
            selected_record.setdefault("grasp_verification", []).append(record)
        print(
            f"GRASP_VERIFY label={label} success={response.success} status={status_name} "
            f"confidence={response.confidence:.2f} wall_s={time.monotonic() - started:.3f} msg={response.message}",
            flush=True,
        )
        for evidence in response.evidence:
            print(f"GRASP_VERIFY_EVIDENCE label={label} value={evidence}", flush=True)
        if not response.success or status != 1:
            raise PickExecutionError(
                f"Grasp retention verification {status_name}: {response.message}",
                phase=f"verify_{label}",
                retryable=False,
            )
        if label == "lift":
            self.current_grasp_verified = True

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

    def recover_after_close_failure(
        self,
        *,
        task_id: str,
        task_desc: str,
        grasp: tuple[float, float, float],
        pregrasp: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float] | None,
    ) -> None:
        retreat = (grasp[0], grasp[1], max(grasp[2], pregrasp[2]))
        candidate = self.current_execution_candidate_index
        recovery_id = f"{task_id}_recover_close_{candidate if candidate is not None else 'unknown'}"
        print(
            f"CLOSE_FAILURE_RECOVERY stage=retreat gripper=closed target={fmt_xyz(retreat)}",
            flush=True,
        )
        ok = self.run_task(
            f"{recovery_id}_retreat",
            f"{task_desc}: retract closed gripper after failed close verification",
            [make_move_step("retreat_closed_gripper_to_pregrasp", retreat, self.args.lift_speed, quat_xyzw)],
        )
        if not ok:
            raise PickExecutionError(
                "Close-failure recovery could not retract to pregrasp",
                phase="recover_close_retreat",
                retryable=False,
            )

        reset_steps = [
            make_gripper_step("open_gripper_after_safe_retreat", 1.0),
            make_wait_step("settle_open_gripper_after_recovery", self.args.open_settle_s),
        ]
        recovery_target = "pregrasp"
        if not self.args.skip_observe:
            observe = (self.args.observe_x, self.args.observe_y, self.args.observe_z)
            reset_steps.extend(
                [
                    make_move_step("return_to_observation_pose", observe, self.args.observe_speed),
                    make_wait_step("settle_recovery_observation_image", self.args.observe_settle_s),
                ]
            )
            recovery_target = "observe"
        ok = self.run_task(
            f"{recovery_id}_reset",
            f"{task_desc}: reset after failed close verification",
            reset_steps,
            timeout_s=90.0,
        )
        if not ok:
            raise PickExecutionError(
                "Close-failure recovery could not open and return to the observation pose",
                phase="recover_close_reset",
                retryable=False,
            )
        print(
            f"CLOSE_FAILURE_RECOVERY success=True final={recovery_target} gripper=open",
            flush=True,
        )

    def recover_after_retention_failure(
        self,
        *,
        task_id: str,
        task_desc: str,
        phase: str,
        elevated_pose: tuple[float, float, float],
    ) -> None:
        print(
            f"RETENTION_FAILURE_RECOVERY phase={phase} stage=open elevated_pose={fmt_xyz(elevated_pose)}",
            flush=True,
        )
        steps = [
            make_gripper_step(f"open_gripper_after_{phase}_failure", 1.0),
            make_wait_step(f"settle_open_gripper_after_{phase}_failure", self.args.open_settle_s),
        ]
        recovery_target = phase
        if not self.args.skip_observe:
            observe = (self.args.observe_x, self.args.observe_y, self.args.observe_z)
            steps.extend(
                [
                    make_move_step("return_to_observation_pose", observe, self.args.observe_speed),
                    make_wait_step("settle_recovery_observation_image", self.args.observe_settle_s),
                ]
            )
            recovery_target = "observe"
        ok = self.run_task(
            f"{task_id}_recover_{phase}",
            f"{task_desc}: reset after {phase} retention failure",
            steps,
            timeout_s=90.0,
        )
        if not ok:
            raise PickExecutionError(
                f"{phase} retention-failure recovery could not open and return to observation",
                phase=f"recover_{phase}",
                retryable=False,
            )
        print(
            f"RETENTION_FAILURE_RECOVERY phase={phase} success=True final={recovery_target} gripper=open",
            flush=True,
        )

    def release_verified_target(
        self,
        *,
        task_id: str,
        task_desc: str,
        current_pose: tuple[float, float, float],
        release_pose: tuple[float, float, float],
        move_quat: tuple[float, float, float, float] | None,
        completion_mode: str,
        drop_height_m: float | None,
    ) -> None:
        need_descent = norm_xyz(sub_xyz(release_pose, current_pose)) > 1e-6
        branch_locked_release = need_descent and self.active_grasp_joint_seed is not None and move_quat is not None
        if branch_locked_release:
            self.run_branch_locked_pose(
                "descend_to_release_height",
                release_pose,
                move_quat,
                self.args.lift_speed,
                phase="release",
                retryable=False,
            )
        release_steps = []
        if need_descent and not branch_locked_release:
            release_steps.append(
                make_move_step("descend_to_release_height", release_pose, self.args.lift_speed, move_quat)
            )
        release_steps.extend(
            [
                make_gripper_step("open_gripper_after_success", 1.0),
                make_wait_step("settle_released_target", max(0.0, self.args.release_settle_s)),
            ]
        )
        ok = self.run_task(
            f"{task_id}_release",
            f"{task_desc}: release after {completion_mode} completion",
            release_steps,
        )
        if not ok:
            raise PickExecutionError(
                "Post-success target release failed",
                phase="release",
                retryable=False,
            )
        selected = next((record for record in self.execution_debug_records if record.get("selected")), None)
        if selected is not None:
            selected["post_success_release"] = {
                "success": True,
                "completion_mode": completion_mode,
                "settle_s": round(max(0.0, float(self.args.release_settle_s)), 3),
                "release_pose": _json_xyz(release_pose),
                "drop_height_m": None if drop_height_m is None else round(drop_height_m, 6),
            }
        print(
            f"POST_SUCCESS_RELEASE success=True completion_mode={completion_mode} "
            f"gripper=open release_pose={fmt_xyz(release_pose)}",
            flush=True,
        )

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
        started = time.monotonic()
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
            f"vol={detection.volume_m3 * 1e6:.1f}cm³ wall_s={time.monotonic() - started:.3f}",
            flush=True,
        )
        return detection

    def request_graspgen_response(
        self,
        debug_output_mode: str | None = None,
    ):
        self.last_graspgen_surface_centroid_camera = None
        self.last_graspgen_volume_centroid_camera = None
        self.last_graspgen_object_top_camera = None
        request = PlanGrasp.Request()
        request.text_prompt = self.args.prompt
        request.confidence_threshold = self.args.confidence_threshold
        request.grasp_threshold = self.args.grasp_threshold
        output_mode = debug_output_mode or self.args.debug_output_mode
        request.debug_output_mode = output_mode

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
        if bool(getattr(response, "success", False)) and int(getattr(response, "object_point_count", 0)) > 0:
            surface = response.object_centroid_xyz
            volume = response.object_volume_centroid_xyz
            self.last_graspgen_surface_centroid_camera = (float(surface.x), float(surface.y), float(surface.z))
            self.last_graspgen_volume_centroid_camera = (float(volume.x), float(volume.y), float(volume.z))
            if bool(getattr(response, "table_plane_found", False)):
                object_top = response.object_top_xyz
                object_top_xyz = (float(object_top.x), float(object_top.y), float(object_top.z))
                if all(math.isfinite(value) for value in object_top_xyz):
                    self.last_graspgen_object_top_camera = object_top_xyz
        self._so101_table_plane = None
        self._so101_table_plane_shadow = None
        self._so101_table_plane_checked = False
        if self.args.so101_tabletop_filter:
            self._cache_so101_table_plane_from_response(response)
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
        target_contact, width_reason, fixed_finger_target_gap = resolve_target_contact_for_candidate(
            self.args, candidate
        )
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
        return (
            t_base_ee,
            t_base_graspgen,
            t_camera_graspgen,
            target_contact,
            adapter_xyz,
            width_reason,
            fixed_finger_target_gap,
        )

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
        target_width_min_offset_m: float | None = None,
        target_width_max_offset_m: float | None = None,
        source_rank_score: float | None = None,
        selection_score: float | None = None,
        fixed_finger_envelope_score_value: float | None = None,
        fixed_finger_gap_score: float | None = None,
        fixed_finger_gap_m: float | None = None,
        fixed_finger_target_gap_m: float | None = None,
        moving_finger_gap_m: float | None = None,
        moving_finger_gap_score: float | None = None,
        fixed_finger_base_side_alignment_cos: float | None = None,
        fixed_finger_inward_offset_m: float | None = None,
        contact_residual_xy_m: float | None = None,
        contact_z_error_m: float | None = None,
        grasp_mesh_min_z: float | None = None,
        so101_tabletop_clearance_m: float | None = None,
        adapter_xyz: Iterable[float] | None = None,
        width_reason: str = "",
        ik_fk_predicted_contact: Iterable[float] | None = None,
        ik_fk_contact_error: Iterable[float] | None = None,
        ik_fk_contact_residual_x: float | None = None,
        ik_fk_contact_residual_y: float | None = None,
        ik_fk_contact_z_error: float | None = None,
        ik_fk_predicted_grasp_mesh_min_z: float | None = None,
        ik_fk_predicted_tabletop_clearance_m: float | None = None,
        ik_grasp_joint5: float | None = None,
        ik_joint5_retry_applied: bool | None = None,
        ik_original_joint5: float | None = None,
        ik_fk_approach_axis_error_deg: float | None = None,
        ik_fk_closing_axis_error_deg: float | None = None,
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
        for key, value in (
            ("target_width_min_offset_m", target_width_min_offset_m),
            ("target_width_max_offset_m", target_width_max_offset_m),
            ("source_rank_score", source_rank_score),
            ("selection_score", selection_score),
            ("fixed_finger_envelope_score", fixed_finger_envelope_score_value),
            ("fixed_finger_gap_score", fixed_finger_gap_score),
            ("fixed_finger_gap_m", fixed_finger_gap_m),
            ("fixed_finger_target_gap_m", fixed_finger_target_gap_m),
            ("moving_finger_gap_m", moving_finger_gap_m),
            ("moving_finger_gap_score", moving_finger_gap_score),
            ("fixed_finger_base_side_alignment_cos", fixed_finger_base_side_alignment_cos),
            ("fixed_finger_inward_offset_m", fixed_finger_inward_offset_m),
            ("contact_residual_xy_m", contact_residual_xy_m),
            ("contact_z_error_m", contact_z_error_m),
            ("ik_grasp_joint5", ik_grasp_joint5),
            ("ik_original_joint5", ik_original_joint5),
            ("ik_fk_approach_axis_error_deg", ik_fk_approach_axis_error_deg),
            ("ik_fk_closing_axis_error_deg", ik_fk_closing_axis_error_deg),
        ):
            if value is not None:
                record[key] = round(float(value), 6)
        if ik_joint5_retry_applied is not None:
            record["ik_joint5_retry_applied"] = bool(ik_joint5_retry_applied)
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
        if ik_fk_predicted_contact is not None:
            record["ik_fk_predicted_contact_base"] = _json_xyz(ik_fk_predicted_contact)
        if ik_fk_contact_error is not None:
            record["ik_fk_contact_error_base"] = _json_xyz(ik_fk_contact_error)
        if ik_fk_contact_residual_x is not None:
            record["ik_fk_contact_residual_x"] = round(float(ik_fk_contact_residual_x), 6)
        if ik_fk_contact_residual_y is not None:
            record["ik_fk_contact_residual_y"] = round(float(ik_fk_contact_residual_y), 6)
        if ik_fk_contact_z_error is not None:
            record["ik_fk_contact_z_error"] = round(float(ik_fk_contact_z_error), 6)
        if ik_fk_predicted_grasp_mesh_min_z is not None:
            record["ik_fk_predicted_grasp_mesh_min_z"] = round(float(ik_fk_predicted_grasp_mesh_min_z), 6)
        if ik_fk_predicted_tabletop_clearance_m is not None:
            record["ik_fk_predicted_tabletop_clearance_m"] = round(float(ik_fk_predicted_tabletop_clearance_m), 6)
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

    def _write_execution_debug_outputs(self, *, render_previews: bool = True) -> None:
        if not self.execution_debug_records:
            return
        out_dir = self.last_graspgen_debug_output_dir
        if out_dir is None:
            return
        started = time.monotonic()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            if self.prepared_candidate_ranking_records:
                (out_dir / "prepared_candidate_ranking.json").write_text(
                    json.dumps(self.prepared_candidate_ranking_records, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            if not self.args.execution_debug_preview:
                return
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
            if self.grasp_verification_records:
                verification_payload = {
                    "service": self.args.grasp_verification_service,
                    "policy": self.args.grasp_verification,
                    "records": self.grasp_verification_records,
                }
                (out_dir / "grasp_verification.json").write_text(
                    json.dumps(verification_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            if not render_previews:
                return
            self._render_execution_debug_preview_svg(out_dir / "grasp_preview_so101_execution.svg")
            self._render_execution_debug_preview_html(out_dir / "grasp_preview_so101_execution.html")
            self._render_execution_stage_overlay_svg(out_dir / "grasp_preview_execution_stages.svg")
            try:
                self._render_execution_debug_preview(out_dir / "grasp_preview_so101_execution.png")
            except Exception as exc:
                print(f"EXECUTION_DEBUG_PNG skipped=True error={exc}", flush=True)
            print(
                f"EXECUTION_DEBUG_OUTPUT dir={out_dir} duration_s={time.monotonic() - started:.3f}",
                flush=True,
            )
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
        if self.last_t_base_camera is not None and self.last_graspgen_object_top_camera is not None:
            top_base = self._transform_cloud(
                self.last_t_base_camera,
                np.asarray([self.last_graspgen_object_top_camera], dtype=np.float64),
            )
            if len(top_base):
                return float(top_base[0, 2]), "plan_grasp_object_top"
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
    @lru_cache(maxsize=4)
    def _preview_control_points(gripper_name: str) -> tuple[np.ndarray, ...]:
        from grasp_gen.robot import load_control_points_for_visualization

        return tuple(
            np.asarray(ctrl_pts, dtype=np.float32) for ctrl_pts in load_control_points_for_visualization(gripper_name)
        )

    @classmethod
    def _preview_lines_for_pose(cls, pose_4x4: np.ndarray, gripper_name: str) -> list[np.ndarray]:
        try:
            lines = []
            for pts in cls._preview_control_points(gripper_name):
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
    @lru_cache(maxsize=4)
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

    @staticmethod
    @lru_cache(maxsize=4)
    def _read_stl_convex_hull_vertices(path: Path, max_triangles: int = 2500) -> np.ndarray:
        vertices, _ = BananaHandeyePickClient._read_stl_triangles(path, max_triangles)
        unique_vertices = np.unique(vertices, axis=0)
        if len(unique_vertices) < 4:
            return unique_vertices
        try:
            return unique_vertices[ConvexHull(unique_vertices).vertices]
        except QhullError:
            return unique_vertices

    @classmethod
    @lru_cache(maxsize=1)
    def _so101_geometry_mesh_data(
        cls,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mesh_dir = Path(__file__).resolve().parents[1] / "src/robot_description/meshes/lerobot/so101"
        fixed_vertices = cls._read_stl_convex_hull_vertices(mesh_dir / "wrist_roll_follower_so101_v1.stl")
        moving_vertices = cls._read_stl_convex_hull_vertices(mesh_dir / "moving_jaw_so101_v1.stl")
        t_gripper_visual = cls._matrix_from_xyz_rpy(
            (5.55112e-17, -0.000218214, 0.000949706),
            (-3.14159, -5.55112e-17, -9.17912e-24),
        )
        t_gripper_jaw = cls._matrix_from_xyz_rpy((0.0202, 0.0188, -0.0234), (1.5708, 0.209440, 0.000001))
        t_jaw_visual = cls._matrix_from_xyz_rpy(
            (-5.55112e-17, -1.94746e-17, 0.0189),
            (9.53145e-17, -4.66093e-24, 0.0),
        )
        return fixed_vertices, moving_vertices, t_gripper_visual, t_gripper_jaw, t_jaw_visual

    @classmethod
    def _so101_width_to_jaw_angle(cls, width_m: float | None) -> float:
        if width_m is None:
            return 0.45
        normalized = (float(width_m) - 0.008) / (0.080 - 0.008)
        return max(0.0, min(1.0, normalized))

    @classmethod
    def _transform_mesh(cls, transform: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        vertices = vertices.astype(np.float64, copy=False)
        return vertices @ transform[:3, :3].T + transform[:3, 3]

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
        return cls._so101_mesh_min_z(meshes)

    @staticmethod
    def _so101_mesh_min_z(meshes: list[tuple[str, np.ndarray, np.ndarray]]) -> float:
        return min(float(vertices[:, 2].min()) for _, vertices, _ in meshes if len(vertices))

    @staticmethod
    def _so101_tabletop_clearance_from_grasp_meshes(
        approach: Iterable[float],
        grasp: Iterable[float],
        meshes: list[tuple[str, np.ndarray, np.ndarray]],
        normal: np.ndarray,
        d: float,
    ) -> float | None:
        min_grasp_clearance = math.inf
        for _, vertices, _ in meshes:
            if len(vertices):
                min_grasp_clearance = min(min_grasp_clearance, float(np.min(vertices @ normal + d)))
        if not math.isfinite(min_grasp_clearance):
            return None

        # Sweep orientation and jaw width are fixed, so signed plane distance is
        # linear in translation and its minimum is at one of the endpoints.
        translation = np.asarray(tuple(approach), dtype=np.float64) - np.asarray(tuple(grasp), dtype=np.float64)
        approach_clearance = min_grasp_clearance + float(translation @ normal)
        return min(min_grasp_clearance, approach_clearance)

    def _so101_gripper_geometry_metrics(
        self,
        approach: tuple[float, float, float],
        grasp: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
        width_m: float | None,
    ) -> tuple[float | None, float | None]:
        try:
            meshes = self._so101_gripper_meshes(grasp, quat_xyzw, width_m)
        except Exception as exc:
            print(f"SO101_MESH_HEIGHT_CHECK skipped=True error={exc}", flush=True)
            if self.args.so101_tabletop_filter:
                print(f"SO101_TABLETOP_FILTER skipped=True reason=mesh_failed error={exc}", flush=True)
            return None, None

        mesh_min_z = self._so101_mesh_min_z(meshes)
        if not self.args.so101_tabletop_filter:
            return mesh_min_z, None
        plane = self._load_so101_table_plane()
        if plane is None:
            return mesh_min_z, None
        normal, d, _ = plane
        clearance = self._so101_tabletop_clearance_from_grasp_meshes(approach, grasp, meshes, normal, d)
        record_shadow = getattr(self, "_record_so101_table_plane_shadow", None)
        if callable(record_shadow):
            record_shadow(approach, grasp, meshes, clearance)
        return mesh_min_z, clearance

    def _so101_gripper_geometry_metrics_batch(
        self,
        candidates: list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float, float],
                float | None,
            ]
        ],
    ) -> list[tuple[float | None, float | None, float | None]]:
        if not candidates:
            return []

        try:
            fixed_vertices, moving_vertices, t_gripper_visual, t_gripper_jaw, t_jaw_visual = (
                self._so101_geometry_mesh_data()
            )
            fixed_transforms = []
            moving_transforms = []
            for _, grasp, quat_xyzw, width_m in candidates:
                t_base_gripper = matrix_from_pose(grasp, quat_xyzw)
                jaw_angle = self._so101_width_to_jaw_angle(width_m)
                t_jaw_motion = self._matrix_from_xyz_rpy((0.0, 0.0, 0.0), (0.0, 0.0, jaw_angle))
                fixed_transforms.append(t_base_gripper @ t_gripper_visual)
                moving_transforms.append(t_base_gripper @ t_gripper_jaw @ t_jaw_motion @ t_jaw_visual)

            fixed_transforms_array = np.stack(fixed_transforms)
            moving_transforms_array = np.stack(moving_transforms)
            fixed_world = (
                np.matmul(
                    fixed_vertices[None, :, :],
                    np.swapaxes(fixed_transforms_array[:, :3, :3], 1, 2),
                )
                + fixed_transforms_array[:, None, :3, 3]
            )
            moving_world = (
                np.matmul(
                    moving_vertices[None, :, :],
                    np.swapaxes(moving_transforms_array[:, :3, :3], 1, 2),
                )
                + moving_transforms_array[:, None, :3, 3]
            )
            min_z = np.minimum(np.min(fixed_world[:, :, 2], axis=1), np.min(moving_world[:, :, 2], axis=1))

            execution_clearance = np.full(len(candidates), np.nan, dtype=np.float64)
            planning_clearance = np.full(len(candidates), np.nan, dtype=np.float64)
            execution_plane = self._load_so101_table_plane() if self.args.so101_tabletop_filter else None
            planning_plane = self._so101_table_plane_shadow if execution_plane is not None else None
            approaches = np.asarray([candidate[0] for candidate in candidates], dtype=np.float64)
            grasps = np.asarray([candidate[1] for candidate in candidates], dtype=np.float64)

            def calculate_clearance(normal: np.ndarray, d: float) -> np.ndarray:
                fixed_grasp = np.min(fixed_world @ normal + d, axis=1)
                moving_grasp = np.min(moving_world @ normal + d, axis=1)
                grasp_clearance = np.minimum(fixed_grasp, moving_grasp)
                approach_clearance = grasp_clearance + (approaches - grasps) @ normal
                return np.minimum(grasp_clearance, approach_clearance)

            if execution_plane is not None:
                normal, d, _ = execution_plane
                execution_clearance = calculate_clearance(normal, d)
            if planning_plane is not None:
                normal, d, _ = planning_plane
                planning_clearance = calculate_clearance(normal, d)

            if execution_plane is not None:
                threshold = float(self.args.so101_tabletop_clearance)
                near_threshold = np.abs(execution_clearance - threshold) <= SO101_GEOMETRY_THRESHOLD_FALLBACK_M
                if planning_plane is not None:
                    near_threshold |= np.abs(planning_clearance - threshold) <= SO101_GEOMETRY_THRESHOLD_FALLBACK_M
                for index in np.flatnonzero(near_threshold):
                    approach, grasp, quat_xyzw, width_m = candidates[int(index)]
                    meshes = self._so101_gripper_meshes(grasp, quat_xyzw, width_m)
                    min_z[index] = self._so101_mesh_min_z(meshes)
                    normal, d, _ = execution_plane
                    exact_execution = self._so101_tabletop_clearance_from_grasp_meshes(
                        approach, grasp, meshes, normal, d
                    )
                    if exact_execution is not None:
                        execution_clearance[index] = exact_execution
                    if planning_plane is not None:
                        normal, d, _ = planning_plane
                        exact_planning = self._so101_tabletop_clearance_from_grasp_meshes(
                            approach, grasp, meshes, normal, d
                        )
                        if exact_planning is not None:
                            planning_clearance[index] = exact_planning

            return [
                (
                    float(min_z[index]),
                    None if not np.isfinite(execution_clearance[index]) else float(execution_clearance[index]),
                    None if not np.isfinite(planning_clearance[index]) else float(planning_clearance[index]),
                )
                for index in range(len(candidates))
            ]
        except Exception as exc:
            print(f"SO101_BATCH_GEOMETRY fallback=True error={exc}", flush=True)
            results = []
            for approach, grasp, quat_xyzw, width_m in candidates:
                min_z, clearance = self._so101_gripper_geometry_metrics(approach, grasp, quat_xyzw, width_m)
                results.append((min_z, clearance, None))
            return results

    @staticmethod
    def _transform_plane(
        transform: np.ndarray,
        normal: Iterable[float],
        d: float,
    ) -> tuple[np.ndarray, float]:
        normal_array = np.asarray(tuple(normal), dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal_array))
        if normal_array.shape != (3,) or not np.isfinite(normal_array).all() or normal_norm <= 1e-9:
            raise ValueError("table plane normal is invalid")
        normal_array /= normal_norm
        d = float(d) / normal_norm
        transformed_normal = transform[:3, :3] @ normal_array
        transformed_d = d - float(transformed_normal @ transform[:3, 3])
        return transformed_normal, transformed_d

    def _cache_so101_table_plane_from_response(self, response) -> bool:
        if not bool(getattr(response, "execution_table_plane_found", False)):
            return False
        if self.last_t_base_camera is None:
            print("SO101_TABLETOP_FILTER response_plane_skipped=True reason=missing_base_camera_transform", flush=True)
            return False
        normal_msg = getattr(response, "execution_table_plane_normal", None)
        if normal_msg is None:
            return False
        try:
            normal, d = self._transform_plane(
                self.last_t_base_camera,
                (float(normal_msg.x), float(normal_msg.y), float(normal_msg.z)),
                float(response.execution_table_plane_offset),
            )
            inlier_ratio = float(response.execution_table_plane_inlier_ratio)
        except (AttributeError, TypeError, ValueError) as exc:
            print(f"SO101_TABLETOP_FILTER response_plane_skipped=True reason=invalid_plane error={exc}", flush=True)
            return False
        if not math.isfinite(d) or not math.isfinite(inlier_ratio):
            print("SO101_TABLETOP_FILTER response_plane_skipped=True reason=non_finite_plane", flush=True)
            return False
        self._so101_table_plane = (normal, d, inlier_ratio)
        if bool(getattr(response, "table_plane_found", False)):
            shadow_normal_msg = getattr(response, "table_plane_normal", None)
            if shadow_normal_msg is not None:
                try:
                    shadow_normal, shadow_d = self._transform_plane(
                        self.last_t_base_camera,
                        (
                            float(shadow_normal_msg.x),
                            float(shadow_normal_msg.y),
                            float(shadow_normal_msg.z),
                        ),
                        float(response.table_plane_offset),
                    )
                    shadow_inlier_ratio = float(response.table_plane_inlier_ratio)
                    if math.isfinite(shadow_d) and math.isfinite(shadow_inlier_ratio):
                        self._so101_table_plane_shadow = (shadow_normal, shadow_d, shadow_inlier_ratio)
                except (AttributeError, TypeError, ValueError) as exc:
                    print(f"SO101_TABLETOP_SHADOW skipped=True reason=invalid_planning_plane error={exc}", flush=True)
        self._so101_table_plane_checked = True
        print(
            "SO101_TABLETOP_FILTER plane_found=True "
            f"normal={fmt_xyz(normal)} d={d:.4f} inlier_ratio={inlier_ratio:.3f} "
            "source=plan_grasp_completed_scene",
            flush=True,
        )
        return True

    def _record_so101_table_plane_shadow(
        self,
        approach: Iterable[float],
        grasp: Iterable[float],
        meshes: list[tuple[str, np.ndarray, np.ndarray]],
        execution_clearance: float | None,
    ) -> None:
        if self._so101_table_plane_shadow is None or execution_clearance is None:
            return
        normal, d, _ = self._so101_table_plane_shadow
        planning_clearance = self._so101_tabletop_clearance_from_grasp_meshes(
            approach,
            grasp,
            meshes,
            normal,
            d,
        )
        if planning_clearance is None:
            return
        self._record_so101_table_plane_shadow_clearances(execution_clearance, planning_clearance)

    def _record_so101_table_plane_shadow_clearances(
        self,
        execution_clearance: float | None,
        planning_clearance: float | None,
    ) -> None:
        if execution_clearance is None or planning_clearance is None:
            return
        threshold = float(self.args.so101_tabletop_clearance)
        execution_passed = execution_clearance >= threshold
        planning_passed = planning_clearance >= threshold
        delta = abs(execution_clearance - planning_clearance)
        self._so101_table_shadow_comparisons += 1
        self._so101_table_shadow_max_clearance_delta = max(self._so101_table_shadow_max_clearance_delta, delta)
        if execution_passed != planning_passed:
            self._so101_table_shadow_mismatches += 1
        print(
            "SO101_TABLETOP_SHADOW "
            f"execution_clearance={execution_clearance:.6f} planning_clearance={planning_clearance:.6f} "
            f"delta_m={delta:.6f} threshold={threshold:.6f} pass_match={execution_passed == planning_passed}",
            flush=True,
        )

    def print_so101_table_plane_shadow_summary(self) -> None:
        if self._so101_table_shadow_comparisons == 0:
            return
        print(
            "SO101_TABLETOP_SHADOW_SUMMARY "
            f"comparisons={self._so101_table_shadow_comparisons} "
            f"mismatches={self._so101_table_shadow_mismatches} "
            f"max_clearance_delta_m={self._so101_table_shadow_max_clearance_delta:.6f}",
            flush=True,
        )

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
            f"normal={fmt_xyz(normal)} d={d:.4f} inlier_ratio={inlier_ratio:.3f} source=debug_ply_fallback",
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
        try:
            meshes = self._so101_gripper_meshes(grasp, quat_xyzw, width_m)
        except Exception as exc:
            print(f"SO101_TABLETOP_FILTER skipped=True reason=mesh_failed error={exc}", flush=True)
            return None
        clearance = self._so101_tabletop_clearance_from_grasp_meshes(approach, grasp, meshes, normal, d)
        record_shadow = getattr(self, "_record_so101_table_plane_shadow", None)
        if callable(record_shadow):
            record_shadow(approach, grasp, meshes, clearance)
        return clearance

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
        selected_records = [record for record in self.execution_debug_records if record.get("selected")]
        selected_record = selected_records[-1] if selected_records else None
        if selected_record is None and self.current_execution_candidate_index is not None:
            failed_record = self._execution_record_for_index(self.current_execution_candidate_index)
            if failed_record is not None and failed_record.get("stage") == "execution_failed":
                selected_record = failed_record
        display_records = [selected_record] if selected_record is not None else []
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
        for record in display_records:
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
        for record in display_records:
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
        if selected_record is not None:
            value = selected_record.get("grasp_contact_base")
            if isinstance(value, list) and len(value) == 3:
                grasp_contact = [float(v) for v in value]
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
        import mpl_toolkits

        matplotlib.use("Agg")
        local_toolkits = Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
        if local_toolkits.is_dir() and str(local_toolkits) not in mpl_toolkits.__path__:
            # Ubuntu's system mpl_toolkits can shadow the venv copy and break mplot3d.
            mpl_toolkits.__path__.insert(0, str(local_toolkits))
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
                for start_point, end_point in zip(path_points, path_points[1:], strict=False):
                    line_segments.append([start_point, end_point])
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

    def target_width_extent_for_candidate(self, candidate):
        if self.last_t_base_camera is None:
            return None
        return target_width_extent_points(
            candidate.pose_matrix,
            self.last_t_base_camera,
            getattr(candidate, "width_axis_camera", (0.0, 0.0, 0.0)),
            float(getattr(candidate, "target_width_min_offset_m", 0.0)),
            float(getattr(candidate, "target_width_max_offset_m", 0.0)),
            float(candidate.target_width_quality),
            float(self.args.target_width_quality_min),
        )

    def fixed_finger_envelope_for_candidate(
        self,
        grasp: tuple[float, float, float],
        quat: tuple[float, float, float, float],
        target_gap_m: float | None,
        extent_points,
    ):
        config = self.args.prepared_candidate_scoring_config
        if (
            not bool(config.get("enabled", False))
            or self.args.target_gripper_type != "asymmetric_single_moving_jaw"
            or self.args.target_fixed_finger_contact_ee is None
            or self.args.target_closing_axis_ee is None
            or target_gap_m is None
            or extent_points is None
        ):
            return None
        return fixed_finger_envelope_score(
            grasp,
            quat,
            self.args.target_fixed_finger_contact_ee,
            self.args.target_closing_axis_ee,
            extent_points[0],
            extent_points[1],
            target_gap_m,
            gap_sigma_m=float(config.get("fixed_finger_gap_sigma_m", 0.006)),
            reliable_max_opening_m=float(config.get("reliable_max_opening_m", 0.072)),
            moving_min_clearance_m=float(config.get("moving_finger_min_clearance_m", 0.003)),
            fixed_score_weight=float(config.get("fixed_finger_score_weight", 0.80)),
        )

    def fixed_finger_base_side_for_candidate(
        self,
        grasp: tuple[float, float, float],
        quat: tuple[float, float, float, float],
        extent_points,
    ):
        config = self.args.target_fixed_finger_base_side_config
        if (
            not bool(config.get("enabled", False))
            or self.args.target_gripper_type != "asymmetric_single_moving_jaw"
            or self.args.target_fixed_finger_contact_ee is None
            or extent_points is None
        ):
            return None
        return fixed_finger_base_side_alignment(
            grasp,
            quat,
            self.args.target_fixed_finger_contact_ee,
            extent_points[0],
            extent_points[1],
            config.get("reference_point_base", (0.0, 0.0, 0.0)),
        )

    def validate_fk_fixed_finger_base_side(
        self,
        label: str,
        ee_xyz: tuple[float, float, float],
        ee_quat_xyzw: tuple[float, float, float, float],
        target_width_extent,
    ):
        config = self.args.target_fixed_finger_base_side_config
        if not bool(config.get("enabled", False)):
            return None
        if target_width_extent is None:
            raise RuntimeError(f"{label}: target width extent is unavailable for FK fixed-finger check")
        try:
            base_side = self.fixed_finger_base_side_for_candidate(
                ee_xyz,
                ee_quat_xyzw,
                target_width_extent,
            )
        except ValueError as exc:
            raise RuntimeError(f"{label}: FK fixed-finger check failed: {exc}") from exc
        if base_side is None:
            raise RuntimeError(f"{label}: FK fixed-finger check is unavailable")
        minimum_alignment = float(config.get("min_alignment_cos", 0.0))
        passed = base_side.alignment_cos >= minimum_alignment
        print(
            f"FK_FIXED_FINGER_SIDE_CHECK label={label} alignment_cos={base_side.alignment_cos:.4f} "
            f"inward_offset_m={base_side.inward_offset_m:.4f} min_alignment_cos={minimum_alignment:.4f} "
            f"passed={passed}",
            flush=True,
        )
        if not passed:
            raise RuntimeError(
                f"{label}: FK fixed finger is on the outer side: alignment={base_side.alignment_cos:.4f} "
                f"< {minimum_alignment:.4f}"
            )
        return base_side

    def rank_graspgen_candidates(
        self,
        candidates,
        base_to_gripper_tf,
    ) -> list[tuple[int, object, float | None, float, float]]:
        indexed = [
            (index, candidate, None, 0.0, float(candidate.confidence)) for index, candidate in enumerate(candidates)
        ]
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
                use_vol = self.args.centroid_source == "volume"
                primary = (
                    self.last_graspgen_volume_centroid_camera if use_vol else self.last_graspgen_surface_centroid_camera
                )
                alternative = (
                    self.last_graspgen_surface_centroid_camera if use_vol else self.last_graspgen_volume_centroid_camera
                )
                if primary is not None:
                    centroid = np.asarray(primary, dtype=np.float64)
                    self.observed_target_base = self.camera_point_to_base(primary, base_to_gripper_tf)
                    self.observed_target_base_alt = (
                        self.camera_point_to_base(alternative, base_to_gripper_tf) if alternative is not None else None
                    )
                    centroid_reason = f"plan_grasp({self.args.centroid_source})"
                    print(
                        f"DETECTION_REUSE source=plan_grasp centroid={fmt_xyz(primary)} "
                        f"centroid_type={self.args.centroid_source}",
                        flush=True,
                    )
                else:
                    detection = self.detect_target()
                    src = detection.volume_centroid_xyz if use_vol else detection.centroid_xyz
                    centroid = np.array([src.x, src.y, src.z], dtype=np.float64)
                    self.observed_target_base = self.detection_to_base(detection, base_to_gripper_tf)
                    self.observed_target_base_alt = self.detection_to_base(
                        detection, base_to_gripper_tf, use_volume=not use_vol
                    )
                    centroid_reason = f"fallback_detection({self.args.centroid_source})"
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
        return [
            (index, candidate, distance, topdown, combined)
            for index, candidate, _, distance, topdown, combined in scored
        ]

    def select_graspgen_candidates(
        self,
        base_to_gripper_tf,
    ) -> list[dict[str, object]]:
        selection_started = time.monotonic()
        self.wait_ik_ready()
        t_base_gripper_start = transform_to_matrix(base_to_gripper_tf)
        start, start_quat = pose_from_matrix(t_base_gripper_start)
        self.last_t_base_camera = t_base_gripper_start @ np.array(self.handeye_matrix, dtype=np.float64)
        stage_started = time.monotonic()
        candidates = self.request_graspgen_candidates(base_to_gripper_tf)
        print(
            f"PIPELINE_TIMING stage=graspgen_request duration_s={time.monotonic() - stage_started:.3f}",
            flush=True,
        )
        self.execution_debug_records = []
        self.prepared_candidate_ranking_records = []
        stage_started = time.monotonic()
        ranked_candidates = self.rank_graspgen_candidates(candidates, base_to_gripper_tf)
        print(
            f"PIPELINE_TIMING stage=candidate_ranking duration_s={time.monotonic() - stage_started:.3f}",
            flush=True,
        )
        max_ik_candidates = int(self.args.max_candidates)
        print(
            f"GRASPGEN_CANDIDATE_TOTAL n={len(candidates)} tested={len(ranked_candidates)} "
            f"max_ik_candidates={max_ik_candidates} ik_filter={self.args.ik_filter} "
            f"ik_check_orientation={self.args.ik_check_orientation} "
            f"ik_fk_contact_compensation={self.use_ik_fk_contact_compensation} "
            f"ik_fk_xy_tolerance={self.args.ik_fk_contact_tolerance:.4f} "
            f"ik_fk_xy_max_correction={self.args.ik_fk_contact_max_correction:.4f} "
            f"ik_fk_z_max_error={self.args.ik_fk_contact_max_xz_error:.4f}",
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
        prepared_scoring = self.args.prepared_candidate_scoring_config
        base_side_config = self.args.target_fixed_finger_base_side_config
        base_side_enabled = bool(base_side_config.get("enabled", False))
        print(
            f"FIXED_FINGER_SOFT_RANK enabled={bool(prepared_scoring.get('enabled', False))} "
            f"envelope_weight={float(prepared_scoring.get('fixed_finger_envelope_weight', 0.55)):.3f} "
            f"gap_sigma={float(prepared_scoring.get('fixed_finger_gap_sigma_m', 0.006)):.4f} "
            f"reliable_max_opening={float(prepared_scoring.get('reliable_max_opening_m', 0.072)):.4f}",
            flush=True,
        )
        print(
            f"FIXED_FINGER_BASE_SIDE enabled={base_side_enabled} "
            f"reference={fmt_xyz(base_side_config.get('reference_point_base', (0.0, 0.0, 0.0)))} "
            f"min_alignment_cos={float(base_side_config.get('min_alignment_cos', 0.0)):.3f}",
            flush=True,
        )
        robust_gap_config = self.args.target_fixed_finger_robust_gap_config
        print(
            f"FIXED_FINGER_ROBUST_GAP enabled={bool(robust_gap_config.get('enabled', False))} "
            f"max_target_gap_deficit="
            f"{float(robust_gap_config.get('max_target_gap_deficit_m', 0.003)):.4f}",
            flush=True,
        )
        orientation_guard = self.args.target_ik_orientation_guard_config
        configured_max_closing = float(
            orientation_guard.get("max_closing_error_deg", JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG)
        )
        effective_max_closing = min(configured_max_closing, JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG)
        print(
            f"IK_ORIENTATION_GUARD enabled={bool(orientation_guard.get('enabled', False))} "
            f"approach_axis={fmt_xyz(orientation_guard.get('approach_axis_ee', (0.0, 0.0, 1.0)))} "
            f"closing_axis_180_symmetric={bool(orientation_guard.get('closing_axis_180_symmetric', False))} "
            f"max_approach_deg={float(orientation_guard.get('max_approach_error_deg', 25.0)):.1f} "
            f"max_closing_deg={effective_max_closing:.1f} "
            f"configured_max_closing_deg={configured_max_closing:.1f}",
            flush=True,
        )

        accepted: list[dict[str, object]] = []
        candidate_geometry_contexts: list[dict[str, object]] = []
        candidate_ik_contexts: list[dict[str, object]] = []
        stage_started = time.monotonic()
        for index, candidate, centroid_dist_camera, topdown_score, source_rank_score in ranked_candidates:
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
                    fixed_finger_target_gap,
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
            target_width_extent = self.target_width_extent_for_candidate(candidate)
            fixed_finger_envelope = self.fixed_finger_envelope_for_candidate(
                grasp,
                quat,
                fixed_finger_target_gap,
                target_width_extent,
            )
            try:
                fixed_finger_base_side = self.fixed_finger_base_side_for_candidate(
                    grasp,
                    quat,
                    target_width_extent,
                )
            except ValueError:
                fixed_finger_base_side = None
            minimum_base_side_alignment = float(base_side_config.get("min_alignment_cos", 0.0))
            if base_side_enabled and (
                fixed_finger_base_side is None or fixed_finger_base_side.alignment_cos < minimum_base_side_alignment
            ):
                alignment_text = (
                    "unavailable" if fixed_finger_base_side is None else f"{fixed_finger_base_side.alignment_cos:.3f}"
                )
                reason = f"fixed_finger_base_side alignment={alignment_text} min={minimum_base_side_alignment:.3f}"
                self._record_execution_candidate(
                    index=index,
                    confidence=confidence,
                    stage="geometry_rejected",
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
                    source_rank_score=source_rank_score,
                    fixed_finger_base_side_alignment_cos=(
                        None if fixed_finger_base_side is None else fixed_finger_base_side.alignment_cos
                    ),
                    fixed_finger_inward_offset_m=(
                        None if fixed_finger_base_side is None else fixed_finger_base_side.inward_offset_m
                    ),
                    adapter_xyz=adapter_xyz,
                    width_reason=width_reason,
                )
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} reason={reason}",
                    flush=True,
                )
                continue
            candidate_geometry_contexts.append(
                {
                    "index": index,
                    "candidate": candidate,
                    "confidence": confidence,
                    "topdown_score": topdown_score,
                    "centroid_dist_camera": centroid_dist_camera,
                    "source_rank_score": source_rank_score,
                    "approach": approach,
                    "grasp": grasp,
                    "lift": lift,
                    "quat": quat,
                    "radius": radius,
                    "contact": contact,
                    "target_contact": target_contact,
                    "execution_contact": execution_contact,
                    "width": width,
                    "width_quality": width_quality,
                    "fixed_finger_envelope": fixed_finger_envelope,
                    "fixed_finger_base_side": fixed_finger_base_side,
                    "target_width_extent": target_width_extent,
                    "adapter_xyz": adapter_xyz,
                    "width_reason": width_reason,
                    "t_base_graspgen": t_base_graspgen,
                    "t_camera_graspgen": t_camera_graspgen,
                }
            )

        geometry_metrics = self._so101_gripper_geometry_metrics_batch(
            [
                (
                    context["approach"],
                    context["grasp"],
                    context["quat"],
                    context["width"],
                )
                for context in candidate_geometry_contexts
            ]
        )
        for context, geometry_metric in zip(candidate_geometry_contexts, geometry_metrics, strict=True):
            index = int(context["index"])
            candidate = context["candidate"]
            confidence = float(context["confidence"])
            topdown_score = float(context["topdown_score"])
            centroid_dist_camera = context["centroid_dist_camera"]
            source_rank_score = float(context["source_rank_score"])
            approach = context["approach"]
            grasp = context["grasp"]
            lift = context["lift"]
            quat = context["quat"]
            radius = float(context["radius"])
            contact = context["contact"]
            target_contact = context["target_contact"]
            execution_contact = context["execution_contact"]
            width = float(context["width"])
            width_quality = float(context["width_quality"])
            fixed_finger_envelope = context["fixed_finger_envelope"]
            fixed_finger_base_side = context["fixed_finger_base_side"]
            target_width_extent = context["target_width_extent"]
            adapter_xyz = context["adapter_xyz"]
            width_reason = str(context["width_reason"])
            t_base_graspgen = context["t_base_graspgen"]
            t_camera_graspgen = context["t_camera_graspgen"]
            grasp_mesh_min_z, so101_tabletop_clearance, planning_tabletop_clearance = geometry_metric
            record_shadow = getattr(self, "_record_so101_table_plane_shadow_clearances", None)
            if callable(record_shadow):
                record_shadow(so101_tabletop_clearance, planning_tabletop_clearance)
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
            if self.args.so101_tabletop_filter and so101_tabletop_clearance is None:
                reason = "so101_tabletop_clearance unavailable while SO101 tabletop filter is enabled"
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

            context["ik_seed"] = self._parallel_ik_seed if self.ik_worker_clients else None
            context["grasp_mesh_min_z"] = grasp_mesh_min_z
            context["so101_tabletop_clearance"] = so101_tabletop_clearance
            candidate_ik_contexts.append(context)

        ik_eligible_count = len(candidate_ik_contexts)
        deferred_ik_contexts: list[dict[str, object]] = []
        if max_ik_candidates > 0 and ik_eligible_count > max_ik_candidates:
            deferred_ik_contexts = candidate_ik_contexts[max_ik_candidates:]
            candidate_ik_contexts = candidate_ik_contexts[:max_ik_candidates]
        for context in deferred_ik_contexts:
            index = int(context["index"])
            candidate = context["candidate"]
            reason = f"max_ik_candidates_{max_ik_candidates}_reached_after_geometry"
            self._record_execution_candidate(
                index=index,
                confidence=float(context["confidence"]),
                stage="ik_deferred",
                reason=reason,
                collision_free=bool(candidate.collision_free),
                topdown_score=float(context["topdown_score"]),
                centroid_dist_camera=context["centroid_dist_camera"],
                approach=context["approach"],
                grasp=context["grasp"],
                lift=context["lift"],
                quat=context["quat"],
                target_contact=context["target_contact"],
                target_width_m=float(context["width"]),
                target_width_quality=float(context["width_quality"]),
                grasp_mesh_min_z=context["grasp_mesh_min_z"],
                so101_tabletop_clearance_m=context["so101_tabletop_clearance"],
                adapter_xyz=context["adapter_xyz"],
                width_reason=str(context["width_reason"]),
            )
            print(
                f"GRASPGEN_CANDIDATE_DEFER idx={index} reason={reason}",
                flush=True,
            )

        print(
            f"PIPELINE_TIMING stage=candidate_geometry duration_s={time.monotonic() - stage_started:.3f} "
            f"candidates={len(candidate_ik_contexts)} eligible={ik_eligible_count} "
            f"deferred={len(deferred_ik_contexts)}",
            flush=True,
        )
        stage_started = time.monotonic()
        ik_results = self._evaluate_candidate_ik_contexts(candidate_ik_contexts)
        print(
            f"PIPELINE_TIMING stage=candidate_ik_fk duration_s={time.monotonic() - stage_started:.3f} "
            f"workers={len(self.ik_worker_clients)} candidates={len(candidate_ik_contexts)}",
            flush=True,
        )
        stage_started = time.monotonic()
        for context, ik_result in zip(candidate_ik_contexts, ik_results, strict=True):
            index = int(context["index"])
            candidate = context["candidate"]
            confidence = float(context["confidence"])
            topdown_score = float(context["topdown_score"])
            centroid_dist_camera = context["centroid_dist_camera"]
            source_rank_score = float(context["source_rank_score"])
            approach = context["approach"]
            grasp = context["grasp"]
            lift = context["lift"]
            quat = context["quat"]
            radius = float(context["radius"])
            target_contact = context["target_contact"]
            execution_contact = context["execution_contact"]
            width = float(context["width"])
            width_quality = float(context["width_quality"])
            fixed_finger_envelope = context["fixed_finger_envelope"]
            fixed_finger_base_side = ik_result["ik_fk_fixed_finger_base_side"] or context["fixed_finger_base_side"]
            grasp_mesh_min_z = context["grasp_mesh_min_z"]
            so101_tabletop_clearance = context["so101_tabletop_clearance"]
            adapter_xyz = context["adapter_xyz"]
            width_reason = str(context["width_reason"])
            failed_reason = str(ik_result["failed_reason"])
            ik_fk_predicted_contact = ik_result["ik_fk_predicted_contact"]
            ik_fk_contact_error = ik_result["ik_fk_contact_error"]
            ik_fk_contact_residual_x = ik_result["ik_fk_contact_residual_x"]
            ik_fk_contact_residual_y = ik_result["ik_fk_contact_residual_y"]
            ik_fk_contact_z_error = ik_result["ik_fk_contact_z_error"]
            ik_fk_predicted_grasp_mesh_min_z = ik_result["ik_fk_predicted_grasp_mesh_min_z"]
            ik_fk_predicted_tabletop_clearance = ik_result["ik_fk_predicted_tabletop_clearance"]
            ik_grasp_joint5 = ik_result["ik_grasp_joint5"]
            ik_joint5_retry_applied = bool(ik_result["ik_joint5_retry_applied"])
            ik_original_joint5 = ik_result["ik_original_joint5"]
            ik_fk_approach_axis_error_deg = ik_result["ik_fk_approach_axis_error_deg"]
            ik_fk_closing_axis_error_deg = ik_result["ik_fk_closing_axis_error_deg"]
            ik_grasp_joint_state = ik_result["ik_grasp_joint_state"]

            if failed_reason:
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
                    ik_fk_predicted_contact=ik_fk_predicted_contact,
                    ik_fk_contact_error=ik_fk_contact_error,
                    ik_fk_contact_residual_x=ik_fk_contact_residual_x,
                    ik_fk_contact_residual_y=ik_fk_contact_residual_y,
                    ik_fk_contact_z_error=ik_fk_contact_z_error,
                    ik_fk_predicted_grasp_mesh_min_z=ik_fk_predicted_grasp_mesh_min_z,
                    ik_fk_predicted_tabletop_clearance_m=ik_fk_predicted_tabletop_clearance,
                    ik_grasp_joint5=ik_grasp_joint5,
                    ik_joint5_retry_applied=ik_joint5_retry_applied,
                    ik_original_joint5=ik_original_joint5,
                    ik_fk_approach_axis_error_deg=ik_fk_approach_axis_error_deg,
                    ik_fk_closing_axis_error_deg=ik_fk_closing_axis_error_deg,
                    fixed_finger_base_side_alignment_cos=(
                        None if fixed_finger_base_side is None else fixed_finger_base_side.alignment_cos
                    ),
                    fixed_finger_inward_offset_m=(
                        None if fixed_finger_base_side is None else fixed_finger_base_side.inward_offset_m
                    ),
                )
                continue

            contact_residual_xy_m = (
                math.hypot(ik_fk_contact_residual_x, ik_fk_contact_residual_y)
                if ik_fk_contact_residual_x is not None and ik_fk_contact_residual_y is not None
                else float(prepared_scoring.get("contact_xy_scale_m", 0.030))
            )
            contact_z_error_m = (
                float(ik_fk_contact_z_error)
                if ik_fk_contact_z_error is not None
                else float(prepared_scoring.get("contact_z_scale_m", 0.020))
            )
            envelope_score = None if fixed_finger_envelope is None else fixed_finger_envelope.score
            selection_score = prepared_candidate_soft_score(
                prepared_scoring,
                fixed_finger_envelope=envelope_score,
                contact_residual_xy_m=contact_residual_xy_m,
                contact_z_error_m=contact_z_error_m,
                confidence=confidence,
                centroid_distance_m=centroid_dist_camera,
            )
            target_width_min_offset_m = float(getattr(candidate, "target_width_min_offset_m", 0.0))
            target_width_max_offset_m = float(getattr(candidate, "target_width_max_offset_m", 0.0))
            envelope_text = "n/a" if envelope_score is None else f"{envelope_score:.3f}"
            base_side_text = "n/a" if fixed_finger_base_side is None else f"{fixed_finger_base_side.alignment_cos:.3f}"
            print(
                f"GRASPGEN_CANDIDATE_ACCEPT idx={index} conf={confidence:.3f} "
                f"approach={fmt_xyz(approach)} grasp={fmt_xyz(grasp)} lift={fmt_xyz(lift)} "
                f"quat={fmt_quat(quat)} selection_score={selection_score:.3f} "
                f"fixed_finger_envelope={envelope_text} fixed_finger_base_side={base_side_text} "
                f"joint5_retry={ik_joint5_retry_applied} "
                f"approach_error_deg={ik_fk_approach_axis_error_deg} "
                f"closing_error_deg={ik_fk_closing_axis_error_deg}",
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
                target_width_min_offset_m=target_width_min_offset_m,
                target_width_max_offset_m=target_width_max_offset_m,
                source_rank_score=source_rank_score,
                selection_score=selection_score,
                fixed_finger_envelope_score_value=envelope_score,
                fixed_finger_gap_score=None if fixed_finger_envelope is None else fixed_finger_envelope.fixed_score,
                fixed_finger_gap_m=None if fixed_finger_envelope is None else fixed_finger_envelope.fixed_gap_m,
                fixed_finger_target_gap_m=(
                    None if fixed_finger_envelope is None else fixed_finger_envelope.target_gap_m
                ),
                moving_finger_gap_m=None if fixed_finger_envelope is None else fixed_finger_envelope.moving_gap_m,
                moving_finger_gap_score=(None if fixed_finger_envelope is None else fixed_finger_envelope.moving_score),
                fixed_finger_base_side_alignment_cos=(
                    None if fixed_finger_base_side is None else fixed_finger_base_side.alignment_cos
                ),
                fixed_finger_inward_offset_m=(
                    None if fixed_finger_base_side is None else fixed_finger_base_side.inward_offset_m
                ),
                contact_residual_xy_m=contact_residual_xy_m,
                contact_z_error_m=contact_z_error_m,
                grasp_mesh_min_z=grasp_mesh_min_z,
                so101_tabletop_clearance_m=so101_tabletop_clearance,
                adapter_xyz=adapter_xyz,
                width_reason=width_reason,
                ik_fk_predicted_contact=ik_fk_predicted_contact,
                ik_fk_contact_error=ik_fk_contact_error,
                ik_fk_contact_residual_x=ik_fk_contact_residual_x,
                ik_fk_contact_residual_y=ik_fk_contact_residual_y,
                ik_fk_contact_z_error=ik_fk_contact_z_error,
                ik_fk_predicted_grasp_mesh_min_z=ik_fk_predicted_grasp_mesh_min_z,
                ik_fk_predicted_tabletop_clearance_m=ik_fk_predicted_tabletop_clearance,
                ik_grasp_joint5=ik_grasp_joint5,
                ik_joint5_retry_applied=ik_joint5_retry_applied,
                ik_original_joint5=ik_original_joint5,
                ik_fk_approach_axis_error_deg=ik_fk_approach_axis_error_deg,
                ik_fk_closing_axis_error_deg=ik_fk_closing_axis_error_deg,
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
                    "target_width_m": width,
                    "target_width_quality": width_quality,
                    "target_width_min_offset_m": target_width_min_offset_m,
                    "target_width_max_offset_m": target_width_max_offset_m,
                    "plan_contact": execution_contact,
                    "ik_fk_predicted_contact": ik_fk_predicted_contact,
                    "ik_fk_contact_error": ik_fk_contact_error,
                    "ik_fk_contact_residual_x": ik_fk_contact_residual_x,
                    "ik_fk_contact_residual_y": ik_fk_contact_residual_y,
                    "ik_fk_contact_z_error": ik_fk_contact_z_error,
                    "contact_residual_xy_m": contact_residual_xy_m,
                    "contact_z_error_m": contact_z_error_m,
                    "centroid_distance_m": centroid_dist_camera,
                    "fixed_finger_envelope": fixed_finger_envelope,
                    "fixed_finger_base_side": fixed_finger_base_side,
                    "target_width_extent": context["target_width_extent"],
                    "selection_score": selection_score,
                    "source_rank_score": source_rank_score,
                    "grasp_joint_seed": ik_grasp_joint_state,
                }
            )

        execution_order = {int(index): position for position, (index, *_rest) in enumerate(ranked_candidates)}
        self.execution_debug_records.sort(
            key=lambda record: execution_order.get(int(record.get("index", -1)), len(execution_order))
        )

        if bool(prepared_scoring.get("enabled", False)) and accepted:
            accepted.sort(
                key=lambda item: (
                    -float(item["selection_score"]),
                    float(item["contact_z_error_m"]),
                    float(item["contact_residual_xy_m"]),
                    -float(item["source_rank_score"]),
                    int(item["index"]),
                )
            )
            order_parts = []
            for item in accepted:
                envelope = item["fixed_finger_envelope"]
                fixed_text = "n/a" if envelope is None else f"{envelope.score:.3f}"
                base_side = item["fixed_finger_base_side"]
                base_side_text = "n/a" if base_side is None else f"{base_side.alignment_cos:.3f}"
                centroid_distance = item["centroid_distance_m"]
                centroid_text = "n/a" if centroid_distance is None else f"{float(centroid_distance):.4f}"
                order_parts.append(
                    f"{int(item['index'])}:score={float(item['selection_score']):.3f}"
                    f"/fixed={fixed_text}/z={float(item['contact_z_error_m']):.4f}"
                    f"/base_side={base_side_text}"
                    f"/dxy={float(item['contact_residual_xy_m']):.4f}"
                    f"/centroid={centroid_text}"
                )
            order = ",".join(order_parts)
            print(f"PREPARED_CANDIDATE_RANK order={order}", flush=True)
        elif self.use_ik_fk_contact_compensation and accepted:
            accepted.sort(
                key=lambda item: (
                    float(item["ik_fk_contact_z_error"]),
                    math.hypot(
                        float(item["ik_fk_contact_residual_x"]),
                        float(item["ik_fk_contact_residual_y"]),
                    ),
                )
            )
            order = ",".join(
                f"{int(item['index'])}:z={float(item['ik_fk_contact_z_error']):.4f}"
                f"/dxy={math.hypot(float(item['ik_fk_contact_residual_x']), float(item['ik_fk_contact_residual_y'])):.4f}"
                for item in accepted
            )
            print(f"IK_FK_CANDIDATE_RANK order={order}", flush=True)

        self.prepared_candidate_ranking_records = []
        for rank, item in enumerate(accepted, start=1):
            envelope = item["fixed_finger_envelope"]
            base_side = item["fixed_finger_base_side"]
            self.prepared_candidate_ranking_records.append(
                {
                    "rank": rank,
                    "candidate_index": int(item["index"]),
                    "selection_score": float(item["selection_score"]),
                    "fixed_finger_envelope_score": None if envelope is None else envelope.score,
                    "fixed_finger_gap_score": None if envelope is None else envelope.fixed_score,
                    "fixed_finger_gap_m": None if envelope is None else envelope.fixed_gap_m,
                    "fixed_finger_target_gap_m": None if envelope is None else envelope.target_gap_m,
                    "moving_finger_gap_m": None if envelope is None else envelope.moving_gap_m,
                    "moving_finger_gap_score": None if envelope is None else envelope.moving_score,
                    "fixed_finger_base_side_alignment_cos": (None if base_side is None else base_side.alignment_cos),
                    "fixed_finger_inward_offset_m": None if base_side is None else base_side.inward_offset_m,
                    "contact_residual_xy_m": float(item["contact_residual_xy_m"]),
                    "contact_z_error_m": float(item["contact_z_error_m"]),
                    "centroid_distance_m": item["centroid_distance_m"],
                    "confidence": float(candidates[int(item["index"])].confidence),
                    "source_rank_score": float(item["source_rank_score"]),
                    "target_width_m": float(item["target_width_m"]),
                    "target_width_quality": float(item["target_width_quality"]),
                    "target_width_min_offset_m": float(item["target_width_min_offset_m"]),
                    "target_width_max_offset_m": float(item["target_width_max_offset_m"]),
                }
            )

        self._write_execution_debug_outputs(render_previews=False)
        print(
            f"PIPELINE_TIMING stage=candidate_finalize duration_s={time.monotonic() - stage_started:.3f}",
            flush=True,
        )
        print(
            f"PIPELINE_TIMING stage=candidate_selection_total duration_s={time.monotonic() - selection_started:.3f}",
            flush=True,
        )
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
        self.selected_target_width_m = float(candidate.get("target_width_m", 0.0))
        self.selected_fixed_finger_envelope = candidate.get("fixed_finger_envelope")
        self.selected_target_width_extent_base = candidate.get("target_width_extent")
        self.selected_grasp_joint_seed = candidate.get("grasp_joint_seed")
        self.active_grasp_joint_seed = self.selected_grasp_joint_seed
        self.current_execution_candidate_index = index
        return (
            candidate["approach"],
            candidate["grasp"],
            candidate["lift"],
            candidate["quat"],
            float(candidate["radius"]),
        )

    def camera_point_to_base(self, camera_point: Iterable[float], base_to_gripper_tf) -> tuple[float, float, float]:
        camera_point = tuple(float(value) for value in camera_point)
        gripper_point = mat4_mul_point(self.handeye_matrix, camera_point)

        t = base_to_gripper_tf.transform.translation
        q = base_to_gripper_tf.transform.rotation
        rotated = quat_rotate((q.x, q.y, q.z, q.w), gripper_point)
        base_point = (rotated[0] + t.x, rotated[1] + t.y, rotated[2] + t.z)

        print(f"TARGET_GRIPPER x={gripper_point[0]:.4f} y={gripper_point[1]:.4f} z={gripper_point[2]:.4f}", flush=True)
        print(f"TARGET_BASE x={base_point[0]:.4f} y={base_point[1]:.4f} z={base_point[2]:.4f}", flush=True)
        return base_point

    def detection_to_base(
        self, detection: Detection2D, base_to_gripper_tf, use_volume: bool | None = None
    ) -> tuple[float, float, float]:
        if use_volume is None:
            use_volume = self.args.centroid_source == "volume"
        src = detection.volume_centroid_xyz if use_volume else detection.centroid_xyz
        return self.camera_point_to_base((src.x, src.y, src.z), base_to_gripper_tf)

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

    def _wait_for_rpc_future(self, future, timeout_s: float, *, background_executor: bool) -> bool:
        if not background_executor:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
            return future.done()

        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if completed.wait(timeout_s):
            return True
        future.cancel()
        return False

    def solve_ik(
        self,
        label: str,
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float] | None = None,
        start_joint_state: JointState | None = None,
        *,
        client=None,
        background_executor: bool = False,
    ) -> tuple[JointState | None, int]:
        client = self.ik_client if client is None else client
        if client is None:
            return None, -1

        request = GetPositionIK.Request()
        request.ik_request.group_name = self.args.ik_group
        request.ik_request.ik_link_name = self.args.ee_frame
        request.ik_request.pose_stamped.header.frame_id = self.args.base_frame
        request.ik_request.pose_stamped.pose = make_pose(*xyz, quat_xyzw)
        request.ik_request.avoid_collisions = bool(self.args.ik_avoid_collisions)
        if start_joint_state is not None:
            request.ik_request.robot_state.joint_state = start_joint_state
        sec, nanosec = self._ik_timeout_duration()
        request.ik_request.timeout.sec = sec
        request.ik_request.timeout.nanosec = nanosec

        future = client.call_async(request)
        rpc_timeout = max(1.0, self.args.ik_timeout_s + 1.0)
        if not self._wait_for_rpc_future(future, rpc_timeout, background_executor=background_executor):
            print(
                f"IK_RESULT label={label} xyz={fmt_xyz(xyz)} quat={fmt_quat(quat_xyzw or (0.0, 0.0, 0.0, 1.0))} ok=False code=timeout",
                flush=True,
            )
            return None, -6

        response = future.result()
        if response is None:
            print(f"IK_RESULT label={label} xyz={fmt_xyz(xyz)} ok=False code=no_response", flush=True)
            return None, -1
        code = int(response.error_code.val)
        ok = code == 1
        print(
            f"IK_RESULT label={label} xyz={fmt_xyz(xyz)} "
            f"quat={fmt_quat(quat_xyzw or (0.0, 0.0, 0.0, 1.0))} ok={ok} code={code}",
            flush=True,
        )
        return (response.solution.joint_state if ok else None), code

    def check_ik(
        self,
        label: str,
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float] | None = None,
    ) -> tuple[bool, int]:
        if not self.args.ik_filter:
            return True, 1
        solution, code = self.solve_ik(label, xyz, quat_xyzw)
        return solution is not None, code

    def compute_fk(
        self,
        label: str,
        joint_state: JointState,
        *,
        client=None,
        background_executor: bool = False,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        client = self.fk_client if client is None else client
        if client is None:
            raise RuntimeError("FK client is disabled")

        request = GetPositionFK.Request()
        request.header.frame_id = self.args.base_frame
        request.fk_link_names = [self.args.ee_frame]
        request.robot_state.joint_state = joint_state
        future = client.call_async(request)
        rpc_timeout = max(1.0, self.args.ik_timeout_s + 1.0)
        if not self._wait_for_rpc_future(future, rpc_timeout, background_executor=background_executor):
            raise RuntimeError(f"FK timed out for {label}")
        response = future.result()
        if response is None:
            raise RuntimeError(f"FK returned no response for {label}")
        code = int(response.error_code.val)
        if code != 1 or not response.pose_stamped:
            raise RuntimeError(f"FK failed for {label}: code={code}")

        pose = response.pose_stamped[0].pose
        xyz = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
        quat = (
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        print(f"FK_RESULT label={label} xyz={fmt_xyz(xyz)} quat={fmt_quat(quat)}", flush=True)
        return xyz, quat

    def validate_grasp_orientation(
        self,
        label: str,
        target_quat_xyzw: tuple[float, float, float, float],
        actual_quat_xyzw: tuple[float, float, float, float],
    ) -> tuple[float | None, float | None]:
        errors = self._grasp_orientation_errors(target_quat_xyzw, actual_quat_xyzw)
        if errors is None:
            return None, None
        config = getattr(self.args, "target_ik_orientation_guard_config", {})
        max_approach = float(config.get("max_approach_error_deg", 25.0))
        max_closing = min(
            float(config.get("max_closing_error_deg", JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG)),
            JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG,
        )
        passed = errors.approach_deg <= max_approach and errors.closing_deg <= max_closing
        print(
            f"IK_ORIENTATION_CHECK label={label} approach_error_deg={errors.approach_deg:.3f} "
            f"closing_error_deg={errors.closing_deg:.3f} max_approach_deg={max_approach:.3f} "
            f"max_closing_deg={max_closing:.3f} passed={passed}",
            flush=True,
        )
        if not passed:
            raise RuntimeError(
                f"FK orientation error exceeds configured limits: approach={errors.approach_deg:.3f}/"
                f"{max_approach:.3f} deg, closing={errors.closing_deg:.3f}/{max_closing:.3f} deg"
            )
        return errors.approach_deg, errors.closing_deg

    def _grasp_orientation_errors(
        self,
        target_quat_xyzw: tuple[float, float, float, float],
        actual_quat_xyzw: tuple[float, float, float, float],
    ):
        config = getattr(self.args, "target_ik_orientation_guard_config", {})
        if not bool(config.get("enabled", False)):
            return None
        closing_axis = getattr(self.args, "target_closing_axis_ee", None)
        if closing_axis is None:
            raise RuntimeError("IK orientation guard requires target_closing_axis_ee")
        return grasp_axis_errors(
            target_quat_xyzw,
            actual_quat_xyzw,
            config.get("approach_axis_ee", (0.0, 0.0, 1.0)),
            closing_axis,
            closing_axis_180_symmetric=bool(config.get("closing_axis_180_symmetric", False)),
        )

    def solve_grasp_ik_fk(
        self,
        label: str,
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
        start_joint_state: JointState | None = None,
        *,
        ik_client=None,
        fk_client=None,
        background_executor: bool = False,
        validate_orientation: bool = True,
    ) -> tuple[IKFKContactPayload | None, int, str]:
        solution, code = self.solve_ik(
            label,
            xyz,
            quat_xyzw,
            start_joint_state,
            client=ik_client,
            background_executor=background_executor,
        )
        if solution is None:
            return None, code, f"ik_failed code={code}"
        solution, original_joint5 = self._apply_joint5_retry_if_needed(
            label,
            xyz,
            quat_xyzw,
            solution,
            ik_client=ik_client,
            background_executor=background_executor,
        )
        if solution is None:
            return None, code, "joint5_bounded_retry_failed"
        try:
            self.validate_grasp_joint5(label, solution)
            ee_xyz, ee_quat = self.compute_fk(
                label,
                solution,
                client=fk_client,
                background_executor=background_executor,
            )
            if validate_orientation:
                approach_error, closing_error = self.validate_grasp_orientation(label, quat_xyzw, ee_quat)
            else:
                errors = self._grasp_orientation_errors(quat_xyzw, ee_quat)
                approach_error = None if errors is None else errors.approach_deg
                closing_error = None if errors is None else errors.closing_deg
        except RuntimeError as exc:
            return None, code, str(exc)
        return (
            IKFKContactPayload(
                joint_state=solution,
                ee_xyz=ee_xyz,
                ee_quat_xyzw=ee_quat,
                joint5_retry_applied=original_joint5 is not None,
                original_joint5=original_joint5,
                approach_axis_error_deg=approach_error,
                closing_axis_error_deg=closing_error,
            ),
            code,
            "",
        )

    @staticmethod
    def _joint_position(joint_state: JointState, joint_name: str) -> float | None:
        positions = dict(zip(joint_state.name, joint_state.position, strict=False))
        value = positions.get(joint_name)
        return None if value is None else float(value)

    @staticmethod
    def _joint_state_with_joint5(joint_state: JointState, joint5: float) -> JointState:
        seed = JointState()
        seed.name = list(joint_state.name)
        seed.position = [
            float(joint5) if str(name) == "5" else float(position)
            for name, position in zip(joint_state.name, joint_state.position, strict=False)
        ]
        return seed

    @staticmethod
    def _canonicalize_joint5(joint5: float) -> float:
        return canonicalize_joint5(joint5)

    def _joint5_closing_axis_correction(
        self,
        target_quat_xyzw: tuple[float, float, float, float],
        actual_quat_xyzw: tuple[float, float, float, float],
    ) -> float:
        config = self.args.target_ik_orientation_guard_config
        closing_axis_ee = getattr(self.args, "target_closing_axis_ee", None)
        if closing_axis_ee is None:
            raise RuntimeError("IK orientation guard requires target_closing_axis_ee")
        try:
            return joint5_closing_axis_correction(
                target_quat_xyzw,
                actual_quat_xyzw,
                config.get("approach_axis_ee", (0.0, 0.0, 1.0)),
                closing_axis_ee,
                closing_axis_180_symmetric=bool(config.get("closing_axis_180_symmetric", False)),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def solve_orientation_consistent_grasp_ik_fk(
        self,
        label: str,
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
        start_joint_state: JointState | None = None,
        *,
        ik_client=None,
        fk_client=None,
        background_executor: bool = False,
    ) -> tuple[IKFKContactPayload | None, int, str]:
        config = self.args.target_ik_orientation_guard_config
        if not bool(config.get("enabled", False)):
            return self.solve_grasp_ik_fk(
                label,
                xyz,
                quat_xyzw,
                start_joint_state,
                ik_client=ik_client,
                fk_client=fk_client,
                background_executor=background_executor,
            )

        max_approach = float(config.get("max_approach_error_deg", 25.0))
        max_closing = min(
            float(config.get("max_closing_error_deg", JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG)),
            JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG,
        )
        seed = start_joint_state
        last_code = -1
        last_reason = "orientation correction was not attempted"
        seen_joint5: set[float] = set()

        for attempt in range(3):
            payload, code, failed_reason = self.solve_grasp_ik_fk(
                f"{label}_orientation_{attempt}",
                xyz,
                quat_xyzw,
                seed,
                ik_client=ik_client,
                fk_client=fk_client,
                background_executor=background_executor,
                validate_orientation=False,
            )
            last_code = code
            if payload is None:
                last_reason = failed_reason
                break

            joint5 = self._joint_position(payload.joint_state, "5")
            approach_error = payload.approach_axis_error_deg
            closing_error = payload.closing_axis_error_deg
            if joint5 is None or approach_error is None or closing_error is None:
                last_reason = "orientation correction requires joint 5 and FK axis errors"
                break
            passed = approach_error <= max_approach and closing_error <= max_closing
            print(
                f"JOINT5_ORIENTATION_SEARCH label={label} attempt={attempt} joint5={joint5:.4f} "
                f"approach_error_deg={approach_error:.3f} closing_error_deg={closing_error:.3f} "
                f"max_approach_deg={max_approach:.3f} max_closing_deg={max_closing:.3f} passed={passed}",
                flush=True,
            )
            if passed:
                return payload, code, ""

            last_reason = (
                f"FK orientation error exceeds strict limits: approach={approach_error:.3f}/{max_approach:.3f} deg, "
                f"closing={closing_error:.3f}/{max_closing:.3f} deg"
            )
            correction = self._joint5_closing_axis_correction(quat_xyzw, payload.ee_quat_xyzw)
            corrected_joint5 = self._canonicalize_joint5(joint5 + correction)
            correction_key = round(corrected_joint5, 9)
            if correction_key in seen_joint5 or abs(corrected_joint5 - joint5) <= 1e-6:
                break
            seen_joint5.add(round(joint5, 9))
            print(
                f"JOINT5_ORIENTATION_CORRECT label={label} attempt={attempt} current={joint5:.4f} "
                f"signed_correction={correction:.4f} corrected={corrected_joint5:.4f}",
                flush=True,
            )
            seed = self._joint_state_with_joint5(payload.joint_state, corrected_joint5)

        return None, last_code, f"no_orientation_consistent_joint5_branch: {last_reason}"

    def _validate_joint5_branch_continuity(
        self,
        label: str,
        seed: JointState | None,
        solution: JointState,
    ) -> None:
        if seed is None or self.args.final_joint5_max is None:
            return
        seed_joint5 = self._joint_position(seed, "5")
        solution_joint5 = self._joint_position(solution, "5")
        if seed_joint5 is None or solution_joint5 is None:
            return
        delta = abs(solution_joint5 - seed_joint5)
        if delta > math.pi / 2.0:
            raise RuntimeError(
                f"{label} changed joint 5 branch by {delta:.4f} rad ({seed_joint5:.4f} -> {solution_joint5:.4f})"
            )

    def prepare_execution_grasp_branch(
        self,
        label: str,
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
        initial_seed: JointState | None,
    ) -> IKFKContactPayload:
        if initial_seed is None:
            raise RuntimeError("Selected grasp candidate has no prepared joint seed")
        initial_joint5 = self._joint_position(initial_seed, "5")
        if initial_joint5 is None:
            raise RuntimeError("Selected grasp candidate seed has no joint 5 position")
        payload, code, failed_reason = self.solve_orientation_consistent_grasp_ik_fk(
            label,
            xyz,
            quat_xyzw,
            initial_seed,
        )
        if payload is None:
            raise RuntimeError(f"Execution branch IK failed: code={code} reason={failed_reason}")
        self.validate_fk_fixed_finger_base_side(
            label,
            payload.ee_xyz,
            payload.ee_quat_xyzw,
            self.selected_target_width_extent_base,
        )
        self.validate_ik_fk_grasp_geometry(
            payload,
            width=self.selected_target_width_m,
            label=f"{label}_branch",
        )
        if self.selected_target_contact_ee is not None and self.selected_plan_contact_base is not None:
            predicted_contact = self._contact_for_pose(
                payload.ee_xyz,
                payload.ee_quat_xyzw,
                self.selected_target_contact_ee,
            )
            _, _, contact_reason = self.ik_fk_contact_guard(
                self.selected_plan_contact_base,
                predicted_contact,
            )
            if contact_reason:
                raise RuntimeError(f"Execution branch contact rejected: {contact_reason}")

        selected_joint5 = self._joint_position(payload.joint_state, "5")
        if selected_joint5 is None:
            raise RuntimeError("Selected execution branch has no joint 5 position")
        self.validate_grasp_joint5(label, payload.joint_state)
        print(
            f"JOINT5_BRANCH_SELECT label={label} initial={initial_joint5:.4f} selected={selected_joint5:.4f} "
            f"half_turn_applied={payload.joint5_retry_applied} "
            f"approach_error_deg={float(payload.approach_axis_error_deg or 0.0):.3f} "
            f"closing_error_deg={float(payload.closing_axis_error_deg or 0.0):.3f}",
            flush=True,
        )
        return payload

    def _evaluate_candidate_ik(
        self,
        context: dict[str, object],
        *,
        ik_client,
        fk_client,
        background_executor: bool,
    ) -> dict[str, object]:
        index = int(context["index"])
        approach = context["approach"]
        grasp = context["grasp"]
        lift = context["lift"]
        quat = context["quat"]
        target_contact = context["target_contact"]
        execution_contact = context["execution_contact"]
        width = float(context["width"])
        ik_seed = context.get("ik_seed")
        target_width_extent = context.get("target_width_extent")

        checks = [("approach", approach)]
        if (
            self.args.require_grasp_ik
            or self.use_ik_fk_contact_compensation
            or self.args.final_joint5_max is not None
            or bool(self.args.target_ik_orientation_guard_config.get("enabled", False))
        ):
            checks.append(("grasp", grasp))
        if self.args.require_lift_ik:
            checks.append(("lift", lift))

        result: dict[str, object] = {
            "failed_reason": "",
            "ik_fk_predicted_contact": None,
            "ik_fk_contact_error": None,
            "ik_fk_contact_residual_x": None,
            "ik_fk_contact_residual_y": None,
            "ik_fk_contact_z_error": None,
            "ik_fk_predicted_grasp_mesh_min_z": None,
            "ik_fk_predicted_tabletop_clearance": None,
            "ik_grasp_joint5": None,
            "ik_joint5_retry_applied": False,
            "ik_original_joint5": None,
            "ik_fk_approach_axis_error_deg": None,
            "ik_fk_closing_axis_error_deg": None,
            "ik_fk_fixed_finger_base_side": None,
            "ik_grasp_joint_state": None,
        }
        for label, xyz in checks:
            needs_grasp_fk = label == "grasp" and (
                self.use_ik_fk_contact_compensation
                or self.args.final_joint5_max is not None
                or bool(self.args.target_ik_orientation_guard_config.get("enabled", False))
            )
            needs_solution = self.args.ik_filter or needs_grasp_fk
            if not needs_solution:
                continue
            ik_quat = quat if self.args.ik_check_orientation or label == "grasp" else None
            if label == "grasp" and needs_grasp_fk:
                payload, code, failed_reason = self.solve_orientation_consistent_grasp_ik_fk(
                    f"graspgen_{index}_{label}",
                    xyz,
                    ik_quat,
                    ik_seed,
                    ik_client=ik_client,
                    fk_client=fk_client,
                    background_executor=background_executor,
                )
                if payload is None:
                    result["failed_reason"] = f"ik_fk_failed_{label} {failed_reason}"
                    print(
                        f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} "
                        f"reason={result['failed_reason']}",
                        flush=True,
                    )
                    return result
                solution = payload.joint_state
                ee_xyz = payload.ee_xyz
                ee_quat = payload.ee_quat_xyzw
                result["ik_grasp_joint5"] = self.validate_grasp_joint5(f"graspgen_{index}_{label}", solution)
                result["ik_joint5_retry_applied"] = payload.joint5_retry_applied
                result["ik_original_joint5"] = payload.original_joint5
                result["ik_fk_approach_axis_error_deg"] = payload.approach_axis_error_deg
                result["ik_fk_closing_axis_error_deg"] = payload.closing_axis_error_deg
                result["ik_grasp_joint_state"] = payload.joint_state
                try:
                    result["ik_fk_fixed_finger_base_side"] = self.validate_fk_fixed_finger_base_side(
                        f"graspgen_{index}_{label}",
                        ee_xyz,
                        ee_quat,
                        target_width_extent,
                    )
                except RuntimeError as exc:
                    result["failed_reason"] = f"ik_fk_fixed_finger_failed_{label} {exc}"
                    print(
                        f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} "
                        f"reason={result['failed_reason']}",
                        flush=True,
                    )
                    return result
            else:
                solution, code = self.solve_ik(
                    f"graspgen_{index}_{label}",
                    xyz,
                    ik_quat,
                    ik_seed,
                    client=ik_client,
                    background_executor=background_executor,
                )
                if solution is None:
                    failed_reason = f"ik_failed_{label} code={code}"
                    result["failed_reason"] = failed_reason
                    print(
                        f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} reason={failed_reason}",
                        flush=True,
                    )
                    return result
            if not self.use_ik_fk_contact_compensation or label != "grasp":
                continue

            predicted_contact = self._contact_for_pose(ee_xyz, ee_quat, target_contact)
            contact_error, contact_z_error, failed_reason = self.ik_fk_contact_guard(
                execution_contact,
                predicted_contact,
            )
            result["ik_fk_predicted_contact"] = predicted_contact
            result["ik_fk_contact_error"] = contact_error
            result["ik_fk_contact_residual_x"] = contact_error[0]
            result["ik_fk_contact_residual_y"] = contact_error[1]
            result["ik_fk_contact_z_error"] = contact_z_error
            if not failed_reason:
                try:
                    mesh_min_z, tabletop_clearance = self.validate_ik_fk_grasp_geometry(
                        payload,
                        width=width,
                        label=f"candidate_{index}",
                    )
                    result["ik_fk_predicted_grasp_mesh_min_z"] = mesh_min_z
                    result["ik_fk_predicted_tabletop_clearance"] = tabletop_clearance
                except RuntimeError as exc:
                    failed_reason = f"ik_fk_geometry_failed_{label} {exc}"
            clearance = result["ik_fk_predicted_tabletop_clearance"]
            clearance_text = "n/a" if clearance is None else f"{float(clearance):.4f}"
            print(
                f"IK_FK_CANDIDATE idx={index} target={fmt_xyz(execution_contact)} "
                f"predicted={fmt_xyz(predicted_contact)} error={fmt_xyz(contact_error)} "
                f"z_error={contact_z_error:.4f} predicted_tabletop_clearance={clearance_text}",
                flush=True,
            )
            if failed_reason:
                result["failed_reason"] = failed_reason
                print(
                    f"GRASPGEN_CANDIDATE_REJECT idx={index} grasp={fmt_xyz(grasp)} reason={failed_reason}",
                    flush=True,
                )
                return result
        return result

    def _evaluate_candidate_ik_contexts(self, contexts: list[dict[str, object]]) -> list[dict[str, object]]:
        if not contexts:
            return []
        if not self.ik_worker_clients:
            return [
                self._evaluate_candidate_ik(
                    context,
                    ik_client=self.ik_client,
                    fk_client=self.fk_client,
                    background_executor=False,
                )
                for context in contexts
            ]

        verification_context = contexts[0]
        verification_xyz = verification_context["approach"]
        verification_quat = verification_context["quat"] if self.args.ik_check_orientation else None
        verification_seed = verification_context.get("ik_seed")
        primary_solution, primary_code = self.solve_ik(
            "worker_verify_primary",
            verification_xyz,
            verification_quat,
            verification_seed,
            client=self.ik_client,
        )
        worker_solution, worker_code = self.solve_ik(
            "worker_verify_0",
            verification_xyz,
            verification_quat,
            verification_seed,
            client=self.ik_worker_clients[0],
        )
        if primary_code != worker_code or (primary_solution is None) != (worker_solution is None):
            raise RuntimeError(
                f"Parallel IK worker verification failed: primary_code={primary_code}, worker_code={worker_code}"
            )
        max_joint_delta = 0.0
        if primary_solution is not None and worker_solution is not None:
            primary_positions = dict(zip(primary_solution.name, primary_solution.position, strict=False))
            worker_positions = dict(zip(worker_solution.name, worker_solution.position, strict=False))
            common_names = sorted(primary_positions.keys() & worker_positions.keys())
            if not common_names:
                raise RuntimeError("Parallel IK worker verification returned no common joints")
            max_joint_delta = max(
                abs(float(primary_positions[name]) - float(worker_positions[name])) for name in common_names
            )
            if max_joint_delta > 1e-8:
                raise RuntimeError(
                    f"Parallel IK worker verification exceeded joint tolerance: max_delta={max_joint_delta:.12f}"
                )
        print(
            f"IK_WORKER_VERIFY success=True code={primary_code} max_joint_delta={max_joint_delta:.12f}",
            flush=True,
        )

        worker_count = min(len(self.ik_worker_clients), len(contexts))
        partitions: list[list[tuple[int, dict[str, object]]]] = [[] for _ in range(worker_count)]
        for position, context in enumerate(contexts):
            partitions[position % worker_count].append((position, context))

        executor = MultiThreadedExecutor(num_threads=max(2, worker_count * 2))
        if not executor.add_node(self):
            raise RuntimeError("Could not attach pick client to the parallel IK executor")
        spin_thread = threading.Thread(target=executor.spin, name="candidate-ik-executor", daemon=True)
        spin_thread.start()
        started = time.monotonic()

        def evaluate_partition(worker_index: int, partition: list[tuple[int, dict[str, object]]]):
            fk_client = self.fk_worker_clients[worker_index] if self.fk_worker_clients else None
            return [
                (
                    position,
                    self._evaluate_candidate_ik(
                        context,
                        ik_client=self.ik_worker_clients[worker_index],
                        fk_client=fk_client,
                        background_executor=True,
                    ),
                )
                for position, context in partition
            ]

        ordered_results: list[dict[str, object] | None] = [None] * len(contexts)
        try:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="candidate-ik-worker") as pool:
                jobs = [pool.submit(evaluate_partition, index, partition) for index, partition in enumerate(partitions)]
                for job in jobs:
                    for position, result in job.result():
                        ordered_results[position] = result
        finally:
            executor.shutdown(timeout_sec=5.0)
            spin_thread.join(timeout=5.0)
            executor.remove_node(self)
        if spin_thread.is_alive():
            raise RuntimeError("Parallel IK executor did not stop cleanly")
        if any(result is None for result in ordered_results):
            raise RuntimeError("Parallel IK worker pool returned incomplete candidate results")
        print(
            f"IK_WORKER_POOL completed=True workers={worker_count} candidates={len(contexts)} "
            f"duration_s={time.monotonic() - started:.3f}",
            flush=True,
        )
        return [result for result in ordered_results if result is not None]

    def predict_contact_from_ik(
        self,
        label: str,
        command_xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
        contact_ee: tuple[float, float, float],
        start_joint_state: JointState | None = None,
    ) -> ContactPrediction[IKFKContactPayload]:
        payload, code, failed_reason = self.solve_grasp_ik_fk(
            label,
            command_xyz,
            quat_xyzw,
            start_joint_state,
        )
        if payload is None:
            raise RuntimeError(f"IK/FK failed for {label}: code={code} reason={failed_reason}")
        self._validate_joint5_branch_continuity(label, start_joint_state, payload.joint_state)
        predicted_contact = self._contact_for_pose(payload.ee_xyz, payload.ee_quat_xyzw, contact_ee)
        return ContactPrediction(
            contact_base=predicted_contact,
            payload=payload,
        )

    def compensate_contact_xy_with_ik_fk(
        self,
        label: str,
        command_xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
        target_contact_base: tuple[float, float, float],
        contact_ee: tuple[float, float, float],
        start_joint_state: JointState | None = None,
    ):
        def predict(
            xyz: tuple[float, float, float],
            previous_payload: IKFKContactPayload | None,
        ) -> ContactPrediction[IKFKContactPayload]:
            return self.predict_contact_from_ik(
                label,
                xyz,
                quat_xyzw,
                contact_ee,
                previous_payload.joint_state if previous_payload is not None else start_joint_state,
            )

        result = compensate_contact_xy(
            command_xyz,
            target_contact_base,
            predict,
            tolerance_m=self.args.ik_fk_contact_tolerance,
            max_iterations=self.args.ik_fk_contact_max_iterations,
            max_correction_m=self.args.ik_fk_contact_max_correction,
        )
        predicted_contact = result.prediction.contact_base
        full_error = sub_xyz(target_contact_base, predicted_contact)
        z_error = abs(full_error[2])
        print(
            f"IK_FK_CONTACT_COMP label={label} converged={result.converged} reason={result.reason} "
            f"solves={result.solve_count} initial_x_error={result.initial_residual_x:.4f} "
            f"initial_y_error={result.initial_residual_y:.4f} "
            f"correction_x={result.correction_x:.4f} correction_y={result.correction_y:.4f} "
            f"residual_x={result.residual_x:.4f} residual_y={result.residual_y:.4f} "
            f"target={fmt_xyz(target_contact_base)} predicted={fmt_xyz(predicted_contact)} "
            f"full_error={fmt_xyz(full_error)} z_error={z_error:.4f} command={fmt_xyz(result.command_xyz)}",
            flush=True,
        )
        return result

    def ik_fk_contact_guard(
        self,
        target_contact_base: tuple[float, float, float],
        predicted_contact_base: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], float, str]:
        error = sub_xyz(target_contact_base, predicted_contact_base)
        z_error = abs(error[2])
        max_correction = float(self.args.ik_fk_contact_max_correction)
        if abs(error[0]) > max_correction or abs(error[1]) > max_correction:
            return (
                error,
                z_error,
                f"ik_fk_contact_xy_residual x={error[0]:.4f} y={error[1]:.4f} exceeds {max_correction:.4f}",
            )
        if z_error > float(self.args.ik_fk_contact_max_xz_error):
            return (
                error,
                z_error,
                f"ik_fk_contact_z_error {z_error:.4f} exceeds {float(self.args.ik_fk_contact_max_xz_error):.4f}",
            )
        if not self.args.allow_out_of_workspace and predicted_contact_base[2] < float(self.args.min_contact_z):
            return (
                error,
                z_error,
                f"ik_fk_predicted_contact_z {predicted_contact_base[2]:.4f} < {float(self.args.min_contact_z):.4f}",
            )
        return error, z_error, ""

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
        planned_contact_base: tuple[float, float, float] | None = None,
    ) -> FixedFingerRobustGap | None:
        if quat_xyzw is None:
            return None
        if planned_contact_base is None:
            planned_contact_base = self.planned_contact_for_pose(grasp, quat_xyzw)
        self.current_plan_contact_base = planned_contact_base
        self.realign_target_contact_base_by_phase["grasp"] = planned_contact_base
        _, correction, error_norm = self.correction_for_contact_alignment(grasp, planned_contact_base)
        xy_error = math.hypot(correction[0], correction[1])
        warn_xy = max(0.0, float(self.args.grasp_realign_max_xy_error))
        realign_xy = float(self.args.grasp_residual_realign_xy_error)
        abort_xy = float(self.args.grasp_residual_abort_xy_error)
        action = "log_only_continue_without_low_height_xy_realign"
        if abort_xy > 0.0 and xy_error > abort_xy:
            action = "log_only_abort_threshold_exceeded"
        elif realign_xy > 0.0 and xy_error > realign_xy:
            action = "log_only_realign_threshold_exceeded"
        elif warn_xy > 0.0 and xy_error > warn_xy:
            action = "warn_continue"
        print(
            f"CONTACT_REALIGN_CHECK phase=grasp error={fmt_xyz(correction)} "
            f"xy_error={xy_error:.4f} error_norm={error_norm:.4f} warn_xy={warn_xy:.4f} "
            f"realign_xy={realign_xy:.4f} abort_xy={abort_xy:.4f} "
            f"planned_contact={fmt_xyz(planned_contact_base)} action={action}",
            flush=True,
        )
        robust_gap = None
        robust_gap_config = self.args.target_fixed_finger_robust_gap_config
        if (
            bool(robust_gap_config.get("enabled", False))
            and self.selected_fixed_finger_envelope is not None
            and self.args.target_closing_axis_ee is not None
        ):
            robust_gap = fixed_finger_robust_gap(
                self.selected_fixed_finger_envelope.fixed_gap_m,
                self.selected_fixed_finger_envelope.target_gap_m,
                correction,
                quat_xyzw,
                self.args.target_closing_axis_ee,
                max_target_gap_deficit_m=float(robust_gap_config.get("max_target_gap_deficit_m", 0.003)),
            )
            selected = next((record for record in self.execution_debug_records if record.get("selected")), None)
            if selected is not None:
                selected["fixed_finger_contact_error_m"] = round(robust_gap.contact_error_along_closing_axis_m, 6)
                selected["fixed_finger_effective_gap_m"] = round(robust_gap.effective_gap_m, 6)
                selected["fixed_finger_required_gap_m"] = round(robust_gap.required_gap_m, 6)
                selected["fixed_finger_robust_gap_passed"] = robust_gap.passed
            print(
                f"FIXED_FINGER_ROBUST_GAP_CHECK passed={robust_gap.passed} "
                f"closing_axis_error={robust_gap.contact_error_along_closing_axis_m:.4f} "
                f"effective_gap={robust_gap.effective_gap_m:.4f} "
                f"required_gap={robust_gap.required_gap_m:.4f}",
                flush=True,
            )
        return robust_gap

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

    def update_selected_ik_fk_compensation(
        self,
        result,
        target_contact_base: tuple[float, float, float],
        predicted_mesh_min_z: float | None,
        predicted_tabletop_clearance: float | None,
    ) -> None:
        selected = next((record for record in self.execution_debug_records if record.get("selected")), None)
        if selected is None:
            return
        payload = result.prediction.payload
        compensation_record: dict[str, object] = {
            "converged": bool(result.converged),
            "reason": result.reason,
            "solve_count": int(result.solve_count),
            "initial_residual_x": round(float(result.initial_residual_x), 6),
            "initial_residual_y": round(float(result.initial_residual_y), 6),
            "correction_x": round(float(result.correction_x), 6),
            "correction_y": round(float(result.correction_y), 6),
            "residual_x": round(float(result.residual_x), 6),
            "residual_y": round(float(result.residual_y), 6),
            "target_contact_base": _json_xyz(target_contact_base),
            "command_grasp": _json_xyz(result.command_xyz),
            "predicted_gripper_base": _json_xyz(payload.ee_xyz),
            "predicted_gripper_quat_xyzw": _json_quat(payload.ee_quat_xyzw),
            "predicted_contact_base": _json_xyz(result.prediction.contact_base),
            "joint_names": [str(name) for name in payload.joint_state.name],
            "joint_positions": [round(float(value), 6) for value in payload.joint_state.position],
            "joint5_retry_applied": bool(payload.joint5_retry_applied),
        }
        if payload.original_joint5 is not None:
            compensation_record["original_joint5"] = round(float(payload.original_joint5), 6)
        if payload.approach_axis_error_deg is not None:
            compensation_record["approach_axis_error_deg"] = round(float(payload.approach_axis_error_deg), 6)
        if payload.closing_axis_error_deg is not None:
            compensation_record["closing_axis_error_deg"] = round(float(payload.closing_axis_error_deg), 6)
        if predicted_mesh_min_z is not None:
            compensation_record["predicted_mesh_min_z"] = round(float(predicted_mesh_min_z), 6)
        if predicted_tabletop_clearance is not None:
            compensation_record["predicted_tabletop_clearance_m"] = round(float(predicted_tabletop_clearance), 6)
        selected["ik_fk_compensation"] = compensation_record

    def _apply_joint5_retry_if_needed(
        self,
        label: str,
        ik_xyz: tuple[float, float, float],
        ik_quat: tuple[float, float, float, float] | None,
        solution: JointState,
        *,
        ik_client,
        background_executor: bool,
    ) -> tuple[JointState | None, float | None]:
        safety_limit = self.args.final_joint5_max
        if safety_limit is None:
            return solution, None
        positions = dict(zip(solution.name, solution.position, strict=False))
        if "5" not in positions:
            return solution, None
        original_j5 = float(positions["5"])
        flip_threshold = math.pi / 2.0
        if abs(original_j5) <= flip_threshold:
            return solution, None

        bounded_seed_j5 = self._canonicalize_joint5(original_j5)
        retry_seed = self._joint_state_with_joint5(solution, bounded_seed_j5)
        retry_solution, retry_code = self.solve_ik(
            f"{label}_joint5_retry",
            ik_xyz,
            ik_quat,
            retry_seed,
            client=ik_client,
            background_executor=background_executor,
        )
        if retry_solution is None:
            print(
                f"JOINT5_RETRY label={label} strategy=half_turn_fixed_finger_swap original={original_j5:.4f} "
                f"original_abs={abs(original_j5):.4f} seed={bounded_seed_j5:.4f} retried=None "
                f"flip_threshold={flip_threshold:.4f} abs_max={float(safety_limit):.4f} "
                f"code={retry_code} status=ik_failed",
                flush=True,
            )
            return None, original_j5
        retry_positions = dict(zip(retry_solution.name, retry_solution.position, strict=False))
        retry_j5 = float(retry_positions.get("5", float("inf")))
        retry_passed = abs(retry_j5) <= float(safety_limit)
        status = "resolved" if retry_passed else "still_outside_abs_limit"
        print(
            f"JOINT5_RETRY label={label} strategy=half_turn_fixed_finger_swap original={original_j5:.4f} "
            f"original_abs={abs(original_j5):.4f} seed={bounded_seed_j5:.4f} retried={retry_j5:.4f} "
            f"retried_abs={abs(retry_j5):.4f} flip_threshold={flip_threshold:.4f} "
            f"abs_max={float(safety_limit):.4f} code={retry_code} status={status}",
            flush=True,
        )
        return (retry_solution if retry_passed else None), original_j5

    def validate_grasp_joint5(self, label: str, joint_state: JointState) -> float | None:
        limit = self.args.final_joint5_max
        if limit is None:
            return None
        positions = dict(zip(joint_state.name, joint_state.position, strict=False))
        if "5" not in positions:
            raise RuntimeError(f"Final joint 5 is unavailable for {label}")
        value = float(positions["5"])
        absolute_value = abs(value)
        passed = absolute_value <= float(limit)
        selected = next((record for record in self.execution_debug_records if record.get("selected")), None)
        if selected is not None:
            selected["final_joint5"] = round(value, 6)
            selected["final_joint5_abs"] = round(absolute_value, 6)
            selected["final_joint5_max"] = round(float(limit), 6)
            selected["final_joint5_abs_max"] = round(float(limit), 6)
            selected["final_joint5_passed"] = passed
        print(
            f"FINAL_JOINT5_CHECK label={label} value={value:.6f} abs={absolute_value:.6f} "
            f"abs_max={float(limit):.6f} passed={passed}",
            flush=True,
        )
        if not passed:
            raise RuntimeError(
                f"Final joint 5 absolute value {absolute_value:.6f} "
                f"exceeds configured maximum {float(limit):.6f} (value={value:.6f})"
            )
        return value

    def validate_ik_fk_grasp_geometry(
        self,
        payload: IKFKContactPayload,
        *,
        width: float | None = None,
        label: str = "final",
    ) -> tuple[float | None, float | None]:
        if width is None:
            selected = next((record for record in self.execution_debug_records if record.get("selected")), None)
            width_value = selected.get("target_width_m") if selected is not None else None
            width = float(width_value) if isinstance(width_value, int | float) else None
        mesh_min_z, tabletop_clearance = self._so101_gripper_geometry_metrics(
            payload.ee_xyz,
            payload.ee_xyz,
            payload.ee_quat_xyzw,
            width,
        )
        clearance_text = "n/a" if tabletop_clearance is None else f"{tabletop_clearance:.4f}"
        mesh_min_z_text = "n/a" if mesh_min_z is None else f"{mesh_min_z:.4f}"
        print(
            f"IK_FK_GEOMETRY_CHECK label={label} mesh_min_z={mesh_min_z_text} tabletop_clearance={clearance_text}",
            flush=True,
        )
        if self.args.so101_tabletop_filter:
            if tabletop_clearance is None:
                raise RuntimeError("SO101 tabletop clearance is unavailable for the IK/FK-predicted grasp")
            if tabletop_clearance < float(self.args.so101_tabletop_clearance):
                raise RuntimeError(
                    f"IK/FK-predicted SO101 tabletop clearance {tabletop_clearance:.4f} "
                    f"< {float(self.args.so101_tabletop_clearance):.4f}"
                )
        return mesh_min_z, tabletop_clearance

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

            if self.active_grasp_joint_seed is not None:
                self.run_branch_locked_pose(
                    f"realign_{phase}_{iteration}",
                    corrected,
                    quat_xyzw,
                    speed,
                    phase=phase,
                )
            else:
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
        target_minus_contact_norm = norm_xyz(target_minus_contact)
        plausibility_limit = max(0.0, float(self.args.pick_diagnostics_max_target_contact_distance))
        plausible = plausibility_limit <= 0.0 or target_minus_contact_norm <= plausibility_limit
        if plausible:
            self.observed_target_base = target_base
        record["target_detection"] = {
            "success": True,
            "plausible": plausible,
            "target_base": _json_xyz(target_base),
            "target_minus_gripper": _json_xyz(target_minus_gripper),
            "target_minus_gripper_norm": round(norm_xyz(target_minus_gripper), 6),
            "target_minus_contact": _json_xyz(target_minus_contact),
            "target_minus_contact_norm": round(target_minus_contact_norm, 6),
            "max_target_contact_distance": round(plausibility_limit, 6),
        }
        self.pick_diagnostic_records.append(record)
        print(
            f"PICK_DIAG_TARGET label={label} success=True plausible={plausible} target_base={fmt_xyz(target_base)} "
            f"target_minus_gripper={fmt_xyz(target_minus_gripper)} gripper_norm={norm_xyz(target_minus_gripper):.4f} "
            f"target_minus_contact={fmt_xyz(target_minus_contact)} contact_norm={target_minus_contact_norm:.4f}",
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

        if (
            self.args.pick_diagnostics
            or self.args.contact_realign
            or self.use_ik_fk_contact_compensation
            or self.use_grasp_verification
        ):
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

            if self.active_grasp_joint_seed is not None and move_quat is not None:
                self.run_branch_locked_pose(
                    "move_above_target",
                    approach,
                    move_quat,
                    self.args.approach_speed,
                    phase="approach",
                )
            else:
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
            if self.active_grasp_joint_seed is not None and move_quat is not None:
                self.run_branch_locked_pose(
                    "move_to_pregrasp_realign_height",
                    pregrasp,
                    move_quat,
                    self.args.descend_speed,
                    phase="pregrasp",
                )
            else:
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

            grasp_diagnostic_xyz = grasp
            grasp_diagnostic_quat = move_quat
            planned_grasp_contact = None
            if self.use_ik_fk_contact_compensation and move_quat is not None:
                if self.selected_plan_contact_base is None or self.selected_target_contact_ee is None:
                    raise PickExecutionError(
                        "IK/FK contact compensation is missing the selected target contact",
                        phase="grasp_compensation",
                        retryable=True,
                    )
                planned_grasp_contact = self.selected_plan_contact_base
                try:
                    result = self.compensate_contact_xy_with_ik_fk(
                        f"graspgen_{self.current_execution_candidate_index}_final",
                        grasp,
                        move_quat,
                        planned_grasp_contact,
                        self.selected_target_contact_ee,
                        self.active_grasp_joint_seed,
                    )
                except RuntimeError as exc:
                    raise PickExecutionError(
                        f"IK/FK contact compensation solver failed: {exc}",
                        phase="grasp_compensation",
                        retryable=True,
                    ) from exc
                if not result.converged:
                    raise PickExecutionError(
                        f"IK/FK contact compensation failed: {result.reason}, "
                        f"residual_x={result.residual_x:.4f} residual_y={result.residual_y:.4f}",
                        phase="grasp_compensation",
                        retryable=True,
                    )

                _, _, contact_guard_reason = self.ik_fk_contact_guard(
                    planned_grasp_contact,
                    result.prediction.contact_base,
                )
                if contact_guard_reason:
                    raise PickExecutionError(
                        f"IK/FK compensated contact rejected: {contact_guard_reason}",
                        phase="grasp_compensation",
                        retryable=True,
                    )

                lift = (lift[0] + result.correction_x, lift[1] + result.correction_y, lift[2])
                grasp = result.command_xyz
                payload = result.prediction.payload
                self.active_grasp_joint_seed = payload.joint_state
                try:
                    self.validate_grasp_joint5(
                        f"graspgen_{self.current_execution_candidate_index}_final",
                        payload.joint_state,
                    )
                    self.validate_fk_fixed_finger_base_side(
                        f"graspgen_{self.current_execution_candidate_index}_final",
                        payload.ee_xyz,
                        payload.ee_quat_xyzw,
                        self.selected_target_width_extent_base,
                    )
                except RuntimeError as exc:
                    self._write_execution_debug_outputs(render_previews=False)
                    raise PickExecutionError(
                        f"Final grasp posture rejected: {exc}",
                        phase="grasp_posture",
                        retryable=True,
                    ) from exc
                try:
                    predicted_mesh_min_z, predicted_tabletop_clearance = self.validate_ik_fk_grasp_geometry(payload)
                except RuntimeError as exc:
                    raise PickExecutionError(
                        f"IK/FK grasp geometry rejected: {exc}",
                        phase="grasp_compensation",
                        retryable=True,
                    ) from exc
                grasp_diagnostic_xyz = payload.ee_xyz
                grasp_diagnostic_quat = payload.ee_quat_xyzw
                self.realign_target_contact_base_by_phase["grasp"] = planned_grasp_contact
                self.current_plan_contact_base = planned_grasp_contact
                self.update_selected_execution_pose(grasp=grasp, lift=lift, quat_xyzw=move_quat)
                self.update_selected_ik_fk_compensation(
                    result,
                    planned_grasp_contact,
                    predicted_mesh_min_z,
                    predicted_tabletop_clearance,
                )
                self._write_execution_debug_outputs(render_previews=False)
                ok = self.run_joint_configuration(
                    "descend_to_ik_fk_compensated_grasp",
                    payload.joint_state,
                    self.args.descend_speed,
                )
            else:
                ok = self.run_task(
                    f"{task_id}_grasp",
                    f"{task_desc}: grasp",
                    [make_move_step("descend_to_graspgen_pose_no_realign", grasp, self.args.descend_speed, move_quat)],
                )
            if not ok:
                raise PickExecutionError("Pick task failed during grasp", phase="grasp", retryable=True)
            robust_gap = self.log_grasp_contact_residual(
                grasp_diagnostic_xyz,
                grasp_diagnostic_quat,
                planned_contact_base=planned_grasp_contact,
            )
            robust_gap_enabled = bool(self.args.target_fixed_finger_robust_gap_config.get("enabled", False))
            if robust_gap_enabled and (robust_gap is None or not robust_gap.passed):
                retreat_ok = self.run_task(
                    f"{task_id}_fixed_finger_gap_retreat",
                    f"{task_desc}: retreat after fixed-finger gap rejection",
                    [make_move_step("retreat_to_pregrasp", pregrasp, self.args.descend_speed, move_quat)],
                )
                if not retreat_ok:
                    raise PickExecutionError(
                        "Fixed-finger robust gap rejected and pregrasp retreat failed",
                        phase="grasp_residual",
                        retryable=False,
                    )
                detail = (
                    "measurement unavailable"
                    if robust_gap is None
                    else (
                        f"effective_gap={robust_gap.effective_gap_m:.4f} "
                        f"required_gap={robust_gap.required_gap_m:.4f} "
                        f"closing_axis_error={robust_gap.contact_error_along_closing_axis_m:.4f}"
                    )
                )
                raise PickExecutionError(
                    f"Fixed-finger robust gap rejected before close: {detail}",
                    phase="grasp_residual",
                    retryable=True,
                )
            self.sample_pick_diagnostics(
                "grasp",
                grasp_diagnostic_xyz,
                commanded_quat_xyzw=grasp_diagnostic_quat,
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
                raise PickExecutionError("Pick task failed during close", phase="close", retryable=False)
            try:
                self.verify_grasp_retention("close")
            except PickExecutionError:
                if self.args.recover_after_close_failure:
                    self.recover_after_close_failure(
                        task_id=task_id,
                        task_desc=task_desc,
                        grasp=grasp,
                        pregrasp=pregrasp,
                        quat_xyzw=move_quat,
                    )
                else:
                    print("CLOSE_FAILURE_RECOVERY enabled=False", flush=True)
                raise
            self.sample_pick_diagnostics(
                "close",
                grasp_diagnostic_xyz,
                commanded_quat_xyzw=grasp_diagnostic_quat,
                detect_target=True,
            )

            probe_height = max(0.0, float(self.args.grasp_verification_probe_lift_height))
            probe_lift = None
            if self.use_grasp_verification and probe_height > 0.0 and lift[2] > grasp[2] + 1e-6:
                probe_z = min(lift[2], grasp[2] + probe_height)
                if probe_z < lift[2] - 1e-6:
                    probe_lift = (lift[0], lift[1], probe_z)
            if probe_lift is not None:
                if self.active_grasp_joint_seed is not None and move_quat is not None:
                    self.run_branch_locked_pose(
                        "probe_lift_target",
                        probe_lift,
                        move_quat,
                        self.args.grasp_verification_probe_lift_speed,
                        phase="probe_lift",
                        retryable=False,
                    )
                else:
                    ok = self.run_task(
                        f"{task_id}_probe_lift",
                        f"{task_desc}: slow retention-check lift",
                        [
                            make_move_step(
                                "probe_lift_target",
                                probe_lift,
                                self.args.grasp_verification_probe_lift_speed,
                                move_quat,
                            )
                        ],
                    )
                    if not ok:
                        raise PickExecutionError(
                            "Pick task failed during probe lift", phase="probe_lift", retryable=False
                        )
                try:
                    self.verify_grasp_retention("probe_lift")
                except PickExecutionError:
                    if self.args.recover_after_retention_failure:
                        self.recover_after_retention_failure(
                            task_id=task_id,
                            task_desc=task_desc,
                            phase="probe_lift",
                            elevated_pose=probe_lift,
                        )
                    else:
                        print("RETENTION_FAILURE_RECOVERY phase=probe_lift enabled=False", flush=True)
                    raise
                self.sample_pick_diagnostics(
                    "probe_lift",
                    probe_lift,
                    commanded_quat_xyzw=move_quat,
                    detect_target=False,
                )

            if self.active_grasp_joint_seed is not None and move_quat is not None:
                self.run_branch_locked_pose(
                    "lift_target",
                    lift,
                    move_quat,
                    self.args.lift_speed,
                    phase="lift",
                    retryable=False,
                    validate_orientation=False,
                )
            else:
                ok = self.run_task(
                    f"{task_id}_lift",
                    f"{task_desc}: lift",
                    [make_move_step("lift_target", lift, self.args.lift_speed, move_quat)],
                )
                if not ok:
                    raise PickExecutionError("Pick task failed during lift", phase="lift", retryable=False)
            try:
                self.verify_grasp_retention("lift")
            except PickExecutionError:
                if self.args.recover_after_retention_failure:
                    self.recover_after_retention_failure(
                        task_id=task_id,
                        task_desc=task_desc,
                        phase="lift",
                        elevated_pose=lift,
                    )
                else:
                    print("RETENTION_FAILURE_RECOVERY phase=lift enabled=False", flush=True)
                raise
            self.sample_pick_diagnostics(
                "lift",
                lift,
                commanded_quat_xyzw=move_quat,
                detect_target=False,
            )
            if self.args.release_after_success:
                release_pose = lift
                release_drop_height = float(self.args.release_drop_height_m)
                if release_drop_height >= 0.0:
                    release_pose = (
                        lift[0],
                        lift[1],
                        min(lift[2], grasp[2] + release_drop_height),
                    )
                need_descent = norm_xyz(sub_xyz(release_pose, lift)) > 1e-6
                branch_locked_release = (
                    need_descent and self.active_grasp_joint_seed is not None and move_quat is not None
                )
                if branch_locked_release:
                    self.run_branch_locked_pose(
                        "descend_to_release_height",
                        release_pose,
                        move_quat,
                        self.args.lift_speed,
                        phase="release",
                        retryable=False,
                    )
                release_steps = []
                if need_descent and not branch_locked_release:
                    release_steps.append(
                        make_move_step("descend_to_release_height", release_pose, self.args.lift_speed, move_quat)
                    )
                release_steps.extend(
                    [
                        make_gripper_step("open_gripper_after_success", 1.0),
                        make_wait_step("settle_released_target", max(0.0, self.args.release_settle_s)),
                    ]
                )
                ok = self.run_task(
                    f"{task_id}_release",
                    f"{task_desc}: release after verified lift",
                    release_steps,
                )
                if not ok:
                    raise PickExecutionError(
                        "Post-success target release failed",
                        phase="release",
                        retryable=False,
                    )
                selected = next((record for record in self.execution_debug_records if record.get("selected")), None)
                if selected is not None:
                    selected["post_success_release"] = {
                        "success": True,
                        "settle_s": round(max(0.0, float(self.args.release_settle_s)), 3),
                        "release_pose": _json_xyz(release_pose),
                        "drop_height_m": None if release_drop_height < 0.0 else round(release_drop_height, 6),
                    }
                print(
                    f"POST_SUCCESS_RELEASE success=True gripper=open release_pose={fmt_xyz(release_pose)}",
                    flush=True,
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
            self.selected_target_width_m = float(candidate.get("target_width_m", 0.0))
            self.selected_fixed_finger_envelope = candidate.get("fixed_finger_envelope")
            self.selected_target_width_extent_base = candidate.get("target_width_extent")
            self.selected_grasp_joint_seed = candidate.get("grasp_joint_seed")
            self.active_grasp_joint_seed = self.selected_grasp_joint_seed
            self.current_grasp_verified = False
            self.realign_target_contact_base_by_phase = {}
            self.mark_execution_candidate_attempt(
                index,
                selected=True,
                stage="selected",
                reason=f"execution_attempt_{attempt}",
            )
            self._write_execution_debug_outputs(render_previews=False)
            print(
                f"GRASPGEN_EXECUTION_ATTEMPT attempt={attempt} idx={index} "
                f"approach={fmt_xyz(candidate['approach'])} grasp={fmt_xyz(candidate['grasp'])}",
                flush=True,
            )
            try:
                if candidate["quat"] is not None and self.selected_grasp_joint_seed is not None:
                    try:
                        branch = self.prepare_execution_grasp_branch(
                            f"graspgen_{index}_execution_branch",
                            candidate["grasp"],
                            candidate["quat"],
                            self.selected_grasp_joint_seed,
                        )
                    except RuntimeError as exc:
                        raise PickExecutionError(
                            f"Could not lock bounded grasp branch: {exc}",
                            phase="grasp_posture",
                            retryable=True,
                        ) from exc
                    self.selected_grasp_joint_seed = branch.joint_state
                    self.active_grasp_joint_seed = branch.joint_state
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
                    reason=(
                        "selected_detect_only"
                        if self.args.detect_only
                        else (
                            f"grasp_verified_successfully_attempt_{attempt}"
                            if self.current_grasp_verified
                            else f"motion_completed_without_verification_attempt_{attempt}"
                        )
                    ),
                )
                return
            except PickExecutionError as exc:
                last_error = exc
                self.mark_execution_candidate_failed(index, exc.phase, str(exc))
                self._write_execution_debug_outputs(render_previews=False)
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
        node.print_so101_table_plane_shadow_summary()
        node._write_execution_debug_outputs()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
