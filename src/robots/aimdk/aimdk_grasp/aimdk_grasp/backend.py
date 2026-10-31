"""The X2 grasp backends, assembled for manipulation_execution's grasp pipeline.

``grasp_execution.backend: "aimdk_grasp.backend:x2_grasp_backends"`` selects
them all in one process; ``aimdk_grasp.remote_head:x2_remote_grasp_backends``
splits them across the pipeline's host and the grasp head's (``aimdk_grasp.split``,
a temporary X2 deployment). The head camera
observes, the perception services (GroundingDINO + SAM2 or grounded SAM2, and
GraspGen-X, on CUDA or the Ascend 310P) propose, cuRobo plans the left arm, the
arm streams with its gravity raise and the left OmniHand grips. The gripper
description is the deployed GraspGen-X bundle's
(``grasp_execution.model_bundle_path``): candidates are generated for it.

``grasp_execution.backend_options`` (all optional):

- ``detect_service``, ``segment_service`` (empty: the detector returns its
  own masks), ``generate_grasps_service``, ``service_timeout_s``
- ``registration_device``, ``depth_scale``, ``calibration_verified``
- ``watch``: ``stop``, ``log`` or ``off``; ``watch_period_s``; ``watch_frame_timeout_s``,
  ``watch_unknown_timeout_s`` and ``watch_image_age_s`` (``execution_watch.WatchTimeouts``)
- ``warmup_prompt``: what the warmup detects before goals are accepted
- ``head``: the grasp head on another host (``aimdk_grasp.split``); remote_head only
- ``perception``: PerceptionSettings fields; ``planning``: GraspRequestConfig fields

``grasp_execution.model_deployment`` must name the GraspGen-X deployment a host
deployment serves (``aimdk_x2_grasp_cuda`` / ``aimdk_x2_grasp_310p``): the shared
``aimdk_x2_grasp`` names none and is refused here.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from manipulation_execution.pick.pipeline.contracts import GraspBackends, GraspFailure
from manipulation_execution.pick.pipeline.perception import PerceptionServices

from .execution_watch import MAX_TIMEOUT_S, WatchTimeouts, finite_seconds
from .feedback import FeedbackHistory
from .grasp_request import GraspRequestConfig, X2GraspPlanner, check_initial_hand
from .proposer import PerceptionSettings, X2Proposer, measured_spheres
from .split import head_options

DEFAULTS = {
    "detect_service": "/perception/grasp/grounding_detect",
    "segment_service": "",
    "generate_grasps_service": "/perception/grasp/generate_grasps",
    # Per model call (the X2 task node's service_timeout_sec).
    "service_timeout_s": 4.0,
    "registration_device": "cuda",
    "depth_scale": -1.0,
    "calibration_verified": True,
    "watch": "stop",
    "watch_period_s": 0.15,
    "watch_frame_timeout_s": WatchTimeouts.frame_s,
    "watch_unknown_timeout_s": WatchTimeouts.unknown_s,
    "watch_image_age_s": WatchTimeouts.image_age_s,
    "warmup_prompt": "a bottle .",
    # The grasp head on another host (aimdk_grasp.split), for remote_head only.
    "head": {},
    "perception": {},
    "planning": {},
}
WARMUP_BUDGET_S = 60.0
#: The runtime streams the X2 backends command. grasp_execution.command_interfaces
#: must list them: robot_config binds them at launch, and only a bound
#: configuration hands the pipeline the runtime's description. What the
#: backends read is checked against that description as they bind it.
COMMAND_INTERFACES = ("joint.arm_stream", "joint.hand_stream")


def _require_declared(grasp_execution: Mapping[str, Any]) -> None:
    missing = sorted(set(COMMAND_INTERFACES) - set(grasp_execution.get("command_interfaces") or ()))
    if missing:
        raise ValueError(f"grasp_execution.command_interfaces must declare the X2 command streams; missing {missing}")


def _options(grasp_execution: Mapping[str, Any]) -> dict[str, Any]:
    given = grasp_execution.get("backend_options") or {}
    unknown = set(given) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"backend_options has unknown keys {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    options = {**DEFAULTS, **given}
    for name, default in DEFAULTS.items():
        value = options[name]
        if isinstance(default, bool | str | dict):
            ok = isinstance(value, type(default))
        else:
            ok = isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        if not ok:
            kind = type(default).__name__ if isinstance(default, bool | str | dict) else "finite number"
            raise ValueError(f"backend_options.{name} must be a {kind}")
    if options["watch"] not in ("stop", "log", "off"):
        raise ValueError("backend_options.watch must be stop, log or off")
    for name in ("watch_frame_timeout_s", "watch_unknown_timeout_s", "watch_image_age_s"):
        if not finite_seconds(options[name], 0.0, MAX_TIMEOUT_S):
            raise ValueError(f"backend_options.{name} must be a finite number of seconds in (0, {MAX_TIMEOUT_S:g}]")
    head_options(options)
    return options


def watch_timeouts(options: Mapping[str, Any]) -> WatchTimeouts:
    return WatchTimeouts(
        frame_s=float(options["watch_frame_timeout_s"]),
        unknown_s=float(options["watch_unknown_timeout_s"]),
        image_age_s=float(options["watch_image_age_s"]),
    )


def _settings(cls, given: Mapping[str, Any], excluded: tuple[str, ...], name: str):
    known = {field.name for field in fields(cls)} - set(excluded)
    unknown = set(given) - known
    if unknown:
        raise ValueError(f"backend_options.{name} has unknown keys {sorted(unknown)}; known: {sorted(known)}")
    return cls(**dict(given))


def _expand(path: str) -> Path:
    """``$(env NAME)`` as the launch files write it."""

    def value(match: re.Match[str]) -> str:
        found = os.environ.get(match.group(1))
        if not found:
            raise ValueError(f"environment variable is not set: {match.group(1)}")
        return found

    return Path(re.sub(r"\$\(env\s+(\w+)\)", value, path))


def gripper_directory(grasp_execution: Mapping[str, Any]) -> Path:
    """The gripper description the deployed GraspGen-X bundle generates candidates for."""
    bundle = str(grasp_execution.get("model_bundle_path", "")).strip()
    if not bundle:
        raise ValueError("grasp_execution.model_bundle_path must name the deployed GraspGen-X bundle")
    root = _expand(bundle)
    assets = json.loads((root / "assets" / "adapter.json").read_text(encoding="utf-8"))
    if assets.get("model_type") != "graspgenx" or not assets.get("gripper_name"):
        raise ValueError(f"{root}: not a GraspGen-X bundle bound to a gripper")
    return root / str(assets.get("gripper_assets_dir", "assets")) / "x_grippers" / str(assets["gripper_name"])


class _ImuPausing:
    """The IMU pauses after each image a request plans from and resumes once there is a plan to execute.

    The watch pauses it again when the execution ends, and so does a request
    whose planning fails: nothing watches the body while the pipeline idles.
    """

    def __init__(self, inner, imu):
        self._inner, self._imu = inner, imu

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _PausingCamera(_ImuPausing):
    def capture(self, **budget):
        observation = self._inner.capture(**budget)
        self._imu.pause_after(observation.stamp_ns)
        return observation


class _ResumingPlanner(_ImuPausing):
    def plan(self, observation, proposals, **budget):
        try:
            planned = self._inner.plan(observation, proposals, **budget)
        except BaseException:
            self._imu.pause()
            raise
        self._imu.resume()
        return planned


@dataclass
class X2Robot:
    """What every host running part of the X2 grasp reads from ``grasp_execution``."""

    options: dict[str, Any]
    perception: PerceptionSettings
    planning: GraspRequestConfig
    gripper: Any
    hand: Any
    gravity: Any
    limits: Any


def x2_robot(grasp_execution: Mapping[str, Any]) -> X2Robot:
    from .arm_gravity import ArmGravity
    from .hand import HandMapping, load_gripper_description
    from .robot_model import arm_limits, planning_model

    _require_declared(grasp_execution)
    if not str(grasp_execution.get("model_deployment") or "").strip():
        raise ValueError(
            "grasp_execution.model_deployment names no GraspGen-X deployment: launch a host deployment of the "
            "X2 grasp (aimdk_x2_grasp_cuda or aimdk_x2_grasp_310p), not the aimdk_x2_grasp they share"
        )
    options = _options(grasp_execution)
    perception = _settings(PerceptionSettings, options["perception"], ("scene",), "perception")
    planning = _settings(GraspRequestConfig, options["planning"], ("revalidation",), "planning")
    model = planning_model()
    gripper = load_gripper_description(gripper_directory(grasp_execution), model)
    hand = HandMapping(model, {name: dict(gripper[name]) for name in ("open", "close")})
    return X2Robot(options, perception, planning, gripper, hand, ArmGravity.load(), arm_limits())


@dataclass
class HeadParts:
    """The X2 head camera, the robot's collision spheres, cuRobo and the target watch:
    what runs beside the head camera and the CUDA GPU."""

    camera: Any
    spheres: Any
    service: Any
    planner: Any
    imu: Any = None
    watch: Any = None
    #: Held by every GPU user of the head (see head_parts).
    gpu: Any = field(default_factory=threading.RLock)


def head_parts(node, robot: X2Robot, description: Mapping[str, Any], closers: list, *, on_moved=None) -> HeadParts:
    """Built on ``node``; each part's close is appended to ``closers``."""
    from .execution_watch import ChestImu, X2TargetWatch
    from .head_camera import HeadCamera
    from .planner import grasp_to_palm_transform
    from .planner_service import GraspPlanningService
    from .robot_spheres import RobotSphereModel

    options, planning = robot.options, robot.planning
    # One lock for every GPU user of the head: while cuRobo captures a CUDA
    # graph, an allocation or synchronization on any other thread of the
    # process fails ("operation not permitted when stream is capturing"). A
    # capture registering with Warp during a planner rebuild did that and
    # took the head down (live 2026-10-22). Re-entrant: the planner's target
    # recheck captures on the planning thread.
    gpu = threading.RLock()
    camera = HeadCamera(
        node,
        description,
        FeedbackHistory(),
        device=options["registration_device"],
        depth_scale=float(options["depth_scale"]),
        calibration_verified=options["calibration_verified"],
        revalidation=planning.revalidation,
        gpu_lock=gpu,
    )
    closers.append(camera.close)
    spheres = RobotSphereModel()
    service = GraspPlanningService(
        gripper=robot.gripper,
        planning_profile=planning.planning_profile,
        approach_seed_policy=planning.approach_seed_policy,
        max_goalset=planning.max_goalset,
        gpu_lock=gpu,
    )
    closers.append(service.close)
    planner = X2GraspPlanner(
        service,
        grasp_to_palm=grasp_to_palm_transform(robot.gripper),
        hand=robot.hand,
        gravity=robot.gravity,
        limits=robot.limits,
        latest_state=camera.capture_state,
        clock=camera.clock,
        config=planning,
    )
    parts = HeadParts(camera, spheres, service, planner, gpu=gpu)
    if options["watch"] != "off":
        parts.imu = ChestImu(node, description)
        closers.append(parts.imu.close)
        parts.watch = X2TargetWatch(
            camera.capture_state,
            lambda feedback: measured_spheres(spheres, feedback),
            mode=options["watch"],
            imu=parts.imu,
            period_s=float(options["watch_period_s"]),
            timeouts=watch_timeouts(options),
            on_moved=on_moved,
            source_clock=camera.clock,
        )
    return parts


def arm_and_hand(node, robot: X2Robot, description: Mapping[str, Any], closers: list):
    """The left arm's streaming executor and the OmniHand effector over the runtime's streams."""
    from manipulation_execution.pick.pipeline.ros_stream import RosStreamIO, bind_stream

    from .arm_stream import x2_arm_executor
    from .omnihand import OmniHandEffector, RosHandIO

    arm = RosStreamIO(
        node,
        bind_stream(
            description,
            stream="joint.arm_stream",
            feedback=["joint.state", "joint.head_state"],
            command="joint.arm_command_state",
        ),
    )
    closers.append(arm.close)
    hand_io = RosHandIO(node, description)
    closers.append(hand_io.close)
    executor = x2_arm_executor(arm, gravity=robot.gravity, limits=robot.limits)
    return executor, OmniHandEffector(hand_io, robot.hand)


def x2_grasp_backends(*, node, grasp_execution: Mapping[str, Any], description: Mapping[str, Any]) -> GraspBackends:
    """Everything in this process (aimdk_grasp.remote_head splits it across two hosts)."""
    from rclpy.callback_groups import ReentrantCallbackGroup

    robot = x2_robot(grasp_execution)
    options = robot.options
    if options["head"]:
        raise ValueError(
            "backend_options.head configures the grasp head of aimdk_grasp.remote_head; "
            "the single-process X2 backends run none"
        )
    closers = []
    try:
        head = head_parts(node, robot, description, closers)
        services = PerceptionServices(
            node,
            detect=options["detect_service"],
            segment=options["segment_service"] or None,
            generate_grasps=options["generate_grasps_service"],
            timeout_s=float(options["service_timeout_s"]),
            callback_group=ReentrantCallbackGroup(),
        )
        proposer = X2Proposer(services, head.spheres, settings=robot.perception, preload=head.service.preload)
        executor, effector = arm_and_hand(node, robot, description, closers)
        source, planner = head.camera, head.planner
        if head.imu is not None:
            source, planner = _PausingCamera(head.camera, head.imu), _ResumingPlanner(head.planner, head.imu)
    except BaseException:
        for close in reversed(closers):
            close()
        raise
    imu = head.imu

    def prepare() -> None:
        """Warm the models and the planner on a fresh image with the hand open, as the X2 did."""
        deadline = time.monotonic() + WARMUP_BUDGET_S
        try:
            observation = source.capture(deadline=deadline, cancelled=lambda: False)
            check_initial_hand(robot.hand, observation.robot_state)
            proposals = proposer.propose(
                observation, options["warmup_prompt"], deadline=deadline, cancelled=lambda: False
            )
            head.service.prepare(observation.robot_state.live.feedback, proposals.scene)
        except GraspFailure as failure:
            raise RuntimeError(f"warmup on {options['warmup_prompt']!r}: {failure}") from failure
        finally:
            if imu is not None:
                imu.pause()  # idle until a request has a plan to execute

    def close() -> None:
        for closer in reversed(closers):
            closer()

    return GraspBackends(source, proposer, planner, executor, effector, head.watch, close=close, prepare=prepare)
