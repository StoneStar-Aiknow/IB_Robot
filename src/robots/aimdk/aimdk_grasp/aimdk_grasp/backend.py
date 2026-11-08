"""The X2 grasp backends, assembled for manipulation_execution's grasp pipeline.

``grasp_execution.backend: "aimdk_grasp.backend:x2_grasp_backends"`` selects
them all in one process; ``aimdk_grasp.remote_head:x2_remote_grasp_backends``
splits them across the pipeline's host and the grasp head's (``aimdk_grasp.split``,
a temporary X2 deployment). The head camera
observes, the perception services (GroundingDINO + SAM2 or grounded SAM2, and
GraspGen-X, on CUDA or the Ascend 310P) propose, cuRobo plans one arm
(``arm_side``, the left by default), the arm streams with its gravity raise and
that side's OmniHand grips. The gripper
description is the deployed GraspGen-X bundle's
(``grasp_execution.model_bundle_path``): candidates are generated for it.

``grasp_execution.backend_options`` (all optional):

- ``arm_side``: ``auto`` in the shipped X2 profiles, or ``left`` / ``right``
  to force an arm. Auto uses torso y (positive left, negative right), trying
  either arm within 5 cm of the midline. Both planners stay resident under
  one GPU lock; execution and the watch follow the successful plan's arm.
  An omitted option preserves the legacy left-arm behavior.
- ``installed_hands``: the OmniHands mounted, ``[left, right]`` by default. A
  robot with one hand removed declares the other alone and forces
  ``arm_side`` to it (auto needs both); the absent hand is then not read, its
  stream slots carry ``joints.ABSENT_HAND_COMMAND``, and it is no obstacle.
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
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from manipulation_execution.grasp_pipeline.contracts import GraspBackends, GraspFailure
from manipulation_execution.grasp_pipeline.perception import PerceptionServices

from .execution_watch import MAX_TIMEOUT_S, WatchTimeouts, finite_seconds
from .feedback import FeedbackHistory
from .grasp_request import GraspRequestConfig, X2GraspPlanner, check_initial_hand
from .joints import installed_hands as parse_installed_hands
from .joints import other_side
from .place_config import PlaceTargetConfig
from .proposer import PerceptionSettings, X2Proposer, measured_spheres
from .split import head_options

DEFAULTS = {
    "arm_side": "left",
    "installed_hands": ["left", "right"],
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
    # Where placement releases the held object in the opening (place_config.PlaceTargetConfig).
    "place": {},
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
        if isinstance(default, list):
            continue  # checked by its own parser below
        if isinstance(default, bool | str | dict):
            ok = isinstance(value, type(default))
        else:
            ok = isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        if not ok:
            kind = type(default).__name__ if isinstance(default, bool | str | dict) else "finite number"
            raise ValueError(f"backend_options.{name} must be a {kind}")
    if options["arm_side"] not in ("left", "right", "auto"):
        raise ValueError("backend_options.arm_side must be left or right or auto")
    hands = installed_hands(options)
    options["installed_hands"] = list(hands)
    if not hands:
        raise ValueError(
            "grasp execution requires at least one installed hand; disable grasp_execution when none are mounted"
        )
    if options["arm_side"] == "auto" and len(hands) != 2:
        raise ValueError(
            f"backend_options.arm_side auto needs both hands; with only {list(hands)} installed, set arm_side to it"
        )
    if options["arm_side"] != "auto" and options["arm_side"] not in hands:
        raise ValueError(
            f"backend_options.arm_side {options['arm_side']!r} names a hand that is not installed ({list(hands)})"
        )
    if options["watch"] not in ("stop", "log", "off"):
        raise ValueError("backend_options.watch must be stop, log or off")
    for name in ("watch_frame_timeout_s", "watch_unknown_timeout_s", "watch_image_age_s"):
        if not finite_seconds(options[name], 0.0, MAX_TIMEOUT_S):
            raise ValueError(f"backend_options.{name} must be a finite number of seconds in (0, {MAX_TIMEOUT_S:g}]")
    head_options(options)
    return options


def installed_hands(options: Mapping[str, Any]) -> tuple[str, ...]:
    """``installed_hands`` of ``backend_options`` checked; both hands when it is omitted."""
    return parse_installed_hands(options.get("installed_hands", DEFAULTS["installed_hands"]))


def grasp_installed_hands(grasp_execution: Mapping[str, Any]) -> tuple[str, ...]:
    """The hands a ``grasp_execution`` configuration declares mounted."""
    return installed_hands(grasp_execution.get("backend_options") or {})


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
    #: The arm (and hand) everything plans and drives: ``backend_options.arm_side``.
    side: str = "left"
    #: Where placement releases the held object: ``backend_options.place``.
    place: PlaceTargetConfig = field(default_factory=PlaceTargetConfig)


def x2_robot(grasp_execution: Mapping[str, Any]) -> X2Robot:
    from .arm_gravity import ArmGravity
    from .hand import HandMapping, load_gripper_description
    from .robot_model import arm_limits, config_directory, planning_model

    _require_declared(grasp_execution)
    if not str(grasp_execution.get("model_deployment") or "").strip():
        raise ValueError(
            "grasp_execution.model_deployment names no GraspGen-X deployment: launch a host deployment of the "
            "X2 grasp (aimdk_x2_grasp_cuda or aimdk_x2_grasp_310p), not the aimdk_x2_grasp they share"
        )
    options = _options(grasp_execution)
    perception = _settings(PerceptionSettings, options["perception"], ("scene",), "perception")
    planning = _settings(GraspRequestConfig, options["planning"], ("revalidation",), "planning")
    place = _settings(PlaceTargetConfig, options["place"], (), "place")
    side = "left" if options["arm_side"] == "auto" else options["arm_side"]
    # The scene is estimated for the planning arm (its reach crop and leg obstacle).
    perception = replace(perception, scene=replace(perception.scene, arm_side=side))
    directory = config_directory(side)
    model = planning_model(directory)
    gripper = load_gripper_description(gripper_directory(grasp_execution), model)
    hand = HandMapping(model, {name: dict(gripper[name]) for name in ("open", "close")}, side=side)
    return X2Robot(
        options,
        perception,
        planning,
        gripper,
        hand,
        ArmGravity.load(side=side),
        arm_limits(directory),
        side,
        place=place,
    )


def robot_for_side(robot: X2Robot, side: str) -> X2Robot:
    """Bind the shared gripper description and settings to one physical arm."""
    from .arm_gravity import ArmGravity
    from .hand import HandMapping
    from .robot_model import arm_limits, config_directory, planning_model

    directory = config_directory(side)
    hand = HandMapping(planning_model(directory), robot.hand.presets, side=side)
    return replace(
        robot,
        side=side,
        hand=hand,
        gravity=ArmGravity.load(side=side),
        limits=arm_limits(directory),
        options={**robot.options, "arm_side": side},
        perception=replace(robot.perception, scene=replace(robot.perception.scene, arm_side=side)),
    )


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


def head_parts(node, robot: X2Robot, description: Mapping[str, Any], closers: list, *, on_moved=None, shared=None):
    """Built on ``node``; each part's close is appended to ``closers``."""
    from .execution_watch import ChestImu, X2TargetWatch
    from .head_camera import HeadCamera
    from .planner import grasp_to_palm_transform
    from .planner_service import GraspPlanningService
    from .robot_model import config_directory
    from .robot_spheres import RobotSphereModel

    if robot.options["arm_side"] == "auto":
        from .dual_arm import AutomaticHead

        robots = {side: robot_for_side(robot, side) for side in ("left", "right")}
        left = head_parts(node, robots["left"], description, closers, on_moved=on_moved)
        right = head_parts(node, robots["right"], description, closers, on_moved=on_moved, shared=left)
        return AutomaticHead({"left": left, "right": right}, robots)

    options, planning = robot.options, robot.planning
    # One lock for every GPU user of the head: while cuRobo captures a CUDA
    # graph, an allocation or synchronization on any other thread of the
    # process fails ("operation not permitted when stream is capturing"). A
    # capture registering with Warp during a planner rebuild did that and
    # took the head down (live 2026-10-22). Re-entrant: the planner's target
    # recheck captures on the planning thread.
    gpu = threading.RLock() if shared is None else shared.gpu
    camera = (
        shared.camera
        if shared is not None
        else HeadCamera(
            node,
            description,
            FeedbackHistory(required_hands=installed_hands(options)),
            device=options["registration_device"],
            depth_scale=float(options["depth_scale"]),
            calibration_verified=options["calibration_verified"],
            revalidation=planning.revalidation,
            gpu_lock=gpu,
        )
    )
    if shared is None:
        closers.append(camera.close)
    spheres = RobotSphereModel(robot.hand, config_directory(robot.side))
    service = GraspPlanningService(
        gripper=robot.gripper,
        side=robot.side,
        planning_profile=planning.planning_profile,
        approach_seed_policy=planning.approach_seed_policy,
        approach_over_height_m=planning.approach_over_height_m,
        approach_order_m=planning.approach_order_m,
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
        parts.imu = ChestImu(node, description) if shared is None else shared.imu
        if shared is None:
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
            side=robot.side,
        )
    return parts


def arm_and_hand(node, robot: X2Robot, description: Mapping[str, Any], closers: list):
    """The ``robot.side`` arm's streaming executor and that side's OmniHand effector over the runtime's streams."""
    from manipulation_execution.grasp_pipeline.ros_stream import RosStreamIO, bind_stream

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
    hand_io = RosHandIO(node, description, robot.side)
    closers.append(hand_io.close)
    executor = x2_arm_executor(arm, gravity=robot.gravity, limits=robot.limits, side=robot.side)
    # The other hand is held where it is measured; one that is not installed is not read at all.
    absent_other = other_side(robot.side) not in installed_hands(robot.options)
    return executor, OmniHandEffector(hand_io, robot.hand, absent_other=absent_other)


def automatic_execution(node, robot, description, closers, planner):
    from .dual_arm import ExecutionSelection, SelectedPart, SelectingPlanner

    pairs = {side: arm_and_hand(node, robot_for_side(robot, side), description, closers) for side in ("left", "right")}
    selection = ExecutionSelection()
    return (
        SelectingPlanner(planner, selection),
        SelectedPart({side: pair[0] for side, pair in pairs.items()}, selection),
        SelectedPart({side: pair[1] for side, pair in pairs.items()}, selection),
    )


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
        source, planner = head.camera, head.planner
        if options["arm_side"] == "auto":

            def local_scene(observation, mask, timeout_s, cancelled):
                if cancelled():
                    raise GraspFailure("CANCELED", "goal cancelled during scene construction")
                prepared = head.prepare_scene(observation)
                scenes = head.complete_scene(prepared, mask, observation)
                head.service.preload(observation.robot_state.live.feedback, scenes)
                return scenes

            proposer = X2Proposer(services, None, settings=robot.perception, remote_scene=local_scene)
            planner, executor, effector = automatic_execution(node, robot, description, closers, planner)
        else:
            proposer = X2Proposer(services, head.spheres, settings=robot.perception, preload=head.service.preload)
            executor, effector = arm_and_hand(node, robot, description, closers)
        if head.imu is not None:
            source, planner = _PausingCamera(head.camera, head.imu), _ResumingPlanner(planner, head.imu)
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
            if options["arm_side"] == "auto":
                head.check_hands(observation.robot_state)  # both arms' hands
                scenes = head.complete_scene(head.prepare_scene(observation), None, observation)
                head.service.prepare(observation.robot_state.live.feedback, scenes)
            else:
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

    from .held_object import held_builder
    from .place_request import X2PlacePlanner

    robots = head.robots if hasattr(head, "arms") else {robot.side: robot}
    held = held_builder(node, executor, effector)
    closers.append(held.close)
    return GraspBackends(
        source,
        proposer,
        planner,
        executor,
        effector,
        head.watch,
        close=close,
        prepare=prepare,
        held_object=held,
        plan_home=X2PlacePlanner(head, robots).plan_home,
    )


def x2_place_backends(*, node, placement_execution, grasp_execution, description):
    return _place_backends(node, grasp_execution, description, placement_execution=placement_execution, remote=False)


def _place_backends(node, grasp_execution, description, *, placement_execution, remote):
    import numpy as np
    from rclpy.callback_groups import ReentrantCallbackGroup

    from manipulation_execution.grasp_pipeline.contracts import PlaceBackends

    from .container_proposer import ContainerProposer
    from .dual_arm import ExecutionSelection, SelectedPart
    from .place_request import X2PlacePlanner
    from .place_verify import PlaceVerifier

    robot = x2_robot(grasp_execution)
    options, closers = robot.options, []
    container_settings = replace(robot.perception, **placement_execution.get("perception", {}))
    try:
        robots = {side: robot_for_side(robot, side) for side in ("left", "right")}
        group = ReentrantCallbackGroup()
        services = PerceptionServices(
            node,
            detect=options["detect_service"],
            segment=options["segment_service"] or None,
            generate_grasps=options["generate_grasps_service"],
            timeout_s=float(options["service_timeout_s"]),
            callback_group=group,
        )
        if remote:
            from .remote_head import RemoteHeadCamera, RemotePlacePlanner, X2GraspHeadClient

            head = X2GraspHeadClient(node, head_options(options).namespace, callback_group=group)
            closers.append(head.close)
            camera, planner = RemoteHeadCamera(head), RemotePlacePlanner(head)
            proposer = ContainerProposer(services, settings=container_settings)
        else:
            if options["head"]:
                raise ValueError("local place backend cannot use a remote head")
            head = head_parts(
                node, replace(robot, options={**options, "arm_side": "auto", "watch": "off"}), description, closers
            )
            camera = head.camera
            planner = X2PlacePlanner(head, robots)
            closers.append(planner.close)
            proposer = ContainerProposer(
                services, models={side: arm.spheres for side, arm in head.arms.items()}, settings=container_settings
            )
        selection = ExecutionSelection()
        pairs = {side: arm_and_hand(node, value, description, closers) for side, value in robots.items()}
        executor = SelectedPart({side: pair[0] for side, pair in pairs.items()}, selection)
        effector = SelectedPart({side: pair[1] for side, pair in pairs.items()}, selection)

        def restore(held):
            executor.finish()
            effector.finish()
            selection.side = held.arm_side
            measured = executor.measured(held.joint_names)
            if held.frame_id != "torso_link" or not np.allclose(measured, held.joint_positions, atol=0.02, rtol=0):
                raise GraspFailure("HELD_OBJECT_MISMATCH", "holding arm differs from the retained handover pose")
            effector.restore(held.effector_state_json)

        def close():
            for closer in reversed(closers):
                closer()

        # The pick's perception settings: find the placed object as the pick found it.
        verifier = PlaceVerifier(services, settings=robot.perception)
        return PlaceBackends(camera, proposer, planner, verifier, executor, effector, restore, close)
    except BaseException:
        for closer in reversed(closers):
            closer()
        raise
