"""Direct-control runtime node: the public runtime contract served for one platform adapter.

One process, two faces:

* Platform side: an adapter owns the transport to the simulator: a server the platform's
  main loop calls each tick (``platform_pull``) or a loop that steps it (``host_step``). Both
  drive the same ``SimulationCore``. The adapter is a ``module:attr`` factory named by the
  platform's embodiment (``adapter`` parameter, never request text), so it belongs to the
  platform registered in ``ibrobot.sim_platforms``; there is no adapter registry of its own.
* ROS side: ``/runtime/*``, ``/motion/{move_to_pose,move_to_joint,compute_ik}``
  (``target_group`` selects the motion group), one gripper FollowJointTrajectory action per
  group, ``/joint_states``, per-group ``ee_pose`` topics plus the default group's
  ``/robot_status/ee_pose``, camera images, and the simulation surface:

  - ``/simulation/cancel_command`` (CancelSimulationCommand): withdraw one command by id;
    ``/simulation/finish_session`` (Trigger): hold position until the platform ends the session.
    ``/simulation/abort_session`` (Trigger): stop writing and end the session without a result.
  - ``/simulation/session`` and ``/simulation/clock``: public lifecycle and tick status. They
    never carry the platform's result, which goes to result listeners only (logged by default).
  - ``/clock``: simulated time, the time of the latest observation over every session of this
    process. It only moves while the platform steps, so it stands still while the runtime is
    idle; consumers on ``use_sim_time`` measure motion in simulated time. This node itself
    runs on wall time (it is the clock source), and stamps every message with simulated time.

Optional layers (an evaluation layer, for one) attach through ``extensions``: ``module:attr``
callables run once with this node after it is built. They may install the engine's motion
gate, add platform-result listeners and serve their own interfaces; the node knows none of them.

Every pose is the tool centre point of its group (the group's ``tcp_offset`` ahead of the
flange pose the platform reports).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from importlib import import_module
from typing import Any

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import (
    CancelSimulationCommand,
    ComputeIk,
    GetRuntimeStatus,
    MoveToConfiguration,
    MoveToPose,
    SetRuntimeMode,
    StopRuntime,
)
from robot_runtime.contract import (
    COMPUTE_IK_SERVICE,
    GET_STATUS_SERVICE,
    IDLE_MODE,
    LIFECYCLE_ACTIVE,
    MOVE_TO_JOINT_SERVICE,
    MOVE_TO_POSE_SERVICE,
    SET_MODE_SERVICE,
    STATUS_TOPIC,
    STOP_POLICIES,
    STOP_SERVICE,
)
from robot_runtime.interface_description import build_description
from robot_runtime.interface_monitor import InterfaceMonitor
from robot_runtime.modes import ModeModel, ModeModelConfig
from robot_runtime.profile import load_profile
from robot_runtime.state import RuntimeState
from robot_runtime_sim.direct_control.engine import (
    OUTCOME_REJECTED,
    OUTCOME_SUCCEEDED,
    Command,
    MotionConfig,
    SimulationCore,
)
from robot_runtime_sim.direct_control.ports import (
    COMMAND_GRIPPER,
    COMMAND_IK,
    COMMAND_JOINT,
    COMMAND_POSE,
    ObservationFrame,
    ik_request_refusal,
    platform_settings,
    profile_mismatches,
)

ABORT_SESSION_SERVICE = "/simulation/abort_session"
IMAGE_REPUBLISH_S = 0.5  # unchanged frames are re-sent at this period (same source stamp)
CANCEL_SERVICE = "/simulation/cancel_command"
FINISH_SESSION_SERVICE = "/simulation/finish_session"
SESSION_TOPIC = "/simulation/session"
CLOCK_TOPIC = "/simulation/clock"
ROS_CLOCK_TOPIC = "/clock"
# Wall-clock bound of one command (a liveness bound, not a motion budget: a stalled platform
# must not hold a caller forever). Below task_executor's 60 s MoveToPose timeout.
MOTION_TIMEOUT_S = 55.0


def resolve_reference(reference: str) -> Any:
    """Import a trusted ``module:attr`` reference (from launch parameters, never from requests)."""
    module_name, separator, attribute = str(reference).strip().partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(f"invalid module:attr reference {reference!r}")
    try:
        return getattr(import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"reference {reference!r} is unresolvable: {exc}") from exc


def ros_time(seconds: float):
    """A builtin_interfaces Time for simulated ``seconds``."""
    return Time(nanoseconds=int(round(max(0.0, seconds) * 1e9))).to_msg()


def _image_msg(rgb, stamp, frame_id: str) -> Image:
    img = Image()
    img.header.stamp = stamp
    img.header.frame_id = frame_id
    img.height, img.width = int(rgb.shape[0]), int(rgb.shape[1])
    img.encoding = "rgb8"
    img.step = img.width * 3
    img.data = rgb.astype("uint8").tobytes()
    return img


def _floats(values) -> list[float]:
    return [float(v) for v in (values.tolist() if hasattr(values, "tolist") else list(values))]


class SimulationRuntime(Node):
    def __init__(self, *, node_name: str = "direct_control_runtime", adapter=None, parameter_overrides=None):
        super().__init__(node_name, parameter_overrides=parameter_overrides or [])
        self.declare_parameter("profile", "")
        self.declare_parameter("instance_id", "")
        # The simulation platform this runtime serves (default: the profile's sim_platform).
        self.declare_parameter("platform", "")
        # ``module:attr`` of the adapter factory ``factory(profile, overrides)``, set by the embodiment.
        self.declare_parameter("adapter", "")
        # JSON object merged into the profile's simulation.<platform>.adapter_config (e.g. a port).
        self.declare_parameter("adapter_overrides_json", "{}")
        # ``module:attr`` callables run once with this node (optional layers, see the module doc).
        self.declare_parameter("extensions", [""])
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("publish_images", True)
        path = str(self.get_parameter("profile").value)
        if not path:
            raise RuntimeError(f"{node_name} needs the 'profile' parameter")
        self._profile = load_profile(path)
        if self.get_parameter("instance_id").value:
            self._profile["runtime"]["instance_id"] = str(self.get_parameter("instance_id").value)
        self._platform = str(self.get_parameter("platform").value or self._profile.get("sim_platform", ""))
        sim = platform_settings(self._profile, self._platform)
        overrides = json.loads(str(self.get_parameter("adapter_overrides_json").value) or "{}")
        if adapter is None:
            factory = str(self.get_parameter("adapter").value or "")
            if not factory:
                raise RuntimeError(f"{node_name} needs the 'adapter' parameter (module:attr of the adapter factory)")
            adapter = resolve_reference(factory)(self._profile, overrides)
        self._adapter = adapter
        descriptor = self._adapter.descriptor
        self._groups: dict[str, dict[str, Any]] = dict(self._profile["motion_groups"])
        if tuple(self._groups) != descriptor.group_names:
            raise RuntimeError(f"profile groups {list(self._groups)} != adapter groups {list(descriptor.group_names)}")
        mismatches = profile_mismatches(self._profile, descriptor)
        if mismatches:
            raise RuntimeError("runtime profile and adapter disagree: " + "; ".join(mismatches))
        self._default_group = str(self._profile.get("default_group") or next(iter(self._groups)))
        self._frame = descriptor.pose_frame
        self._camera_prefix = str(self._profile.get("camera_topic_prefix", "/simulation")).rstrip("/")
        description = build_description(
            self._profile,
            use_sim=bool(self._profile.get("use_sim", False)),
            sim_platform=self._platform if self._profile.get("use_sim") else None,
        )
        self._interface_monitor = InterfaceMonitor(self, description)
        self._modes = ModeModel(ModeModelConfig.from_profile(self._profile))
        self._state = RuntimeState(
            str(self._profile["runtime"]["name"]),
            str(self._profile["runtime"]["version"]),
            dict(self._profile["capabilities"]),
            self._modes,
            on_change=self._publish_status,
            interface_description=description,
            interface_states=self._interface_monitor.states,
        )

        latched = QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
        )
        self._session_pub = self.create_publisher(String, SESSION_TOPIC, latched)
        self._clock_pub = self.create_publisher(String, CLOCK_TOPIC, 10)
        self._ros_clock_pub = self.create_publisher(Clock, ROS_CLOCK_TOPIC, 10)
        self._result_listeners: list[Callable[[dict], None]] = []
        self._core = SimulationCore(
            descriptor,
            MotionConfig.from_mapping(sim.get("motion")),
            on_event=self._on_core_event,
            on_platform_result=self._on_platform_result,
            on_clock=self._publish_ros_clock,
        )
        self._last_image_source: tuple | None = None
        self._last_image_sent = 0.0
        self._image_msgs: dict[str, Image] = {}
        self._last_clock: tuple | None = None

        cb = ReentrantCallbackGroup()
        self._status_pub = self.create_publisher(RuntimeStatus, STATUS_TOPIC, 10)
        self._joint_pub = self.create_publisher(JointState, str(self._profile["joint_state_topic"]), 10)
        self._ee_pubs = {
            group: self.create_publisher(PoseStamped, str(spec["ee_pose_topic"]), 10)
            for group, spec in self._groups.items()
        }
        self._default_ee_pub = self.create_publisher(PoseStamped, "/robot_status/ee_pose", 10)
        self._image_pubs = (
            {
                cam: self.create_publisher(Image, f"{self._camera_prefix}/{cam}/image_raw", 2)
                for cam in descriptor.cameras
            }
            if bool(self.get_parameter("publish_images").value)
            else {}
        )
        self._overlay_pubs = {
            cam: self.create_publisher(Image, f"{self._camera_prefix}/{cam}/overlay/image_raw", 2)
            for cam in getattr(self._adapter, "overlay_cameras", ())
            if cam in self._image_pubs
        }

        self.create_service(SetRuntimeMode, SET_MODE_SERVICE, self._on_set_mode, callback_group=cb)
        self.create_service(GetRuntimeStatus, GET_STATUS_SERVICE, self._on_get_status, callback_group=cb)
        self.create_service(StopRuntime, STOP_SERVICE, self._on_stop, callback_group=cb)
        if descriptor.supports(COMMAND_POSE):
            self.create_service(MoveToPose, MOVE_TO_POSE_SERVICE, self._on_move_to_pose, callback_group=cb)
        if descriptor.supports(COMMAND_JOINT) or descriptor.supports(COMMAND_GRIPPER):
            self.create_service(MoveToConfiguration, MOVE_TO_JOINT_SERVICE, self._on_move_to_joint, callback_group=cb)
        if descriptor.supports(COMMAND_IK):
            self.create_service(ComputeIk, COMPUTE_IK_SERVICE, self._on_compute_ik, callback_group=cb)
        self.create_service(CancelSimulationCommand, CANCEL_SERVICE, self._on_cancel_command, callback_group=cb)
        self.create_service(Trigger, FINISH_SESSION_SERVICE, self._on_finish_session, callback_group=cb)
        self.create_service(Trigger, ABORT_SESSION_SERVICE, self._on_abort_session, callback_group=cb)
        self._gripper_servers = []
        for group, spec in self._groups.items():
            if not spec.get("gripper") or not descriptor.supports(COMMAND_GRIPPER):
                continue
            name = f"/{spec['gripper']}_trajectory_controller/follow_joint_trajectory"
            self._gripper_servers.append(
                ActionServer(
                    self,
                    FollowJointTrajectory,
                    name,
                    execute_callback=lambda handle, g=group: self._on_gripper(g, handle),
                    goal_callback=lambda request, g=group: self._on_gripper_goal(g, request),
                    cancel_callback=lambda _h: CancelResponse.ACCEPT,  # honoured in _on_gripper
                    callback_group=cb,
                )
            )

        for reference in [str(r).strip() for r in self.get_parameter("extensions").value if str(r).strip()]:
            resolve_reference(reference)(self)
            self.get_logger().info(f"extension {reference} attached")

        self._adapter.start(self._core, self.get_logger())
        timer_cb = MutuallyExclusiveCallbackGroup()
        # Wall-time timers (this node does not use sim time): state and status keep flowing
        # while the simulated clock stands still.
        self.create_timer(1.0 / float(self.get_parameter("publish_rate").value), self._publish_state, timer_cb)
        self.create_timer(1.0, self._publish_status, callback_group=cb)
        self._publish_ros_clock(self._core.sim_time_s())
        self._state.set_lifecycle(LIFECYCLE_ACTIVE)

    @property
    def core(self) -> SimulationCore:
        return self._core

    @property
    def adapter(self):
        return self._adapter

    def add_platform_result_listener(self, listener: Callable[[dict], None]) -> None:
        """Hear each session's platform result (an extension's hook; the node only logs it)."""
        self._result_listeners.append(listener)

    # ------------------------------------------------------------------ helpers

    def _resolve_group(self, requested: str) -> tuple[str | None, str]:
        group = requested.strip() or self._default_group
        if group not in self._groups:
            return None, f"unknown target_group {requested!r}; this runtime has {sorted(self._groups)}"
        return group, ""

    def _motion_refusal(self) -> str:
        if self._state.stop_latched:
            return "runtime is STOPPED (stop latched)"
        if not self._modes.spec().allows_trajectory:
            return f"mode {self._modes.mode!r} does not permit trajectory execution"
        return ""

    def _on_core_event(self, kind: str, info: dict) -> None:
        payload = {"event": kind, "wall_time": time.time(), **info}
        self._session_pub.publish(String(data=json.dumps(payload, default=str)))
        if kind != "state":
            self.get_logger().info(f"simulation {kind}: {json.dumps(info, default=str)[:300]}")

    def _on_platform_result(self, result: dict) -> None:
        # Not published: a platform's verdict must not reach the robot's consumers (the Agent).
        # Whoever is entitled to it registers a listener (an evaluation layer, for one).
        self.get_logger().info(
            f"platform result of {result.get('session_id')}: success={result.get('native_success')} "
            f"reason={result.get('failure_reason')} interrupted={result.get('interrupted')}"
        )
        for listener in list(self._result_listeners):
            try:
                listener(dict(result))
            except Exception as exc:  # one listener must not break the session end
                self.get_logger().error(f"platform result listener failed: {exc!r}")

    # ------------------------------------------------------------------ publishing

    def _sim_now(self):
        return ros_time(self._core.sim_time_s())

    def _publish_ros_clock(self, seconds: float) -> None:
        msg = Clock()
        msg.clock = ros_time(seconds)
        self._ros_clock_pub.publish(msg)

    def _publish_status(self) -> None:
        self._status_pub.publish(self._state.to_msg(self._sim_now()))

    def _publish_state(self) -> None:
        try:
            self._publish_state_unchecked()
        except Exception as exc:  # one bad observation must not stop the state stream
            self.get_logger().error(f"state publishing failed: {exc!r}", throttle_duration_sec=5.0)

    def _publish_clock(self) -> None:
        # Late subscribers of /clock get the current (possibly standing) simulated time too.
        self._publish_ros_clock(self._core.sim_time_s())
        clock = self._core.clock()
        key = (clock["session_id"], clock["generation"], clock["tick"], clock["state"])
        clock["wall_time"] = time.time()
        if key != self._last_clock or int(clock["wall_time"] * 2) % 2 == 0:
            self._last_clock = key
            self._clock_pub.publish(String(data=json.dumps(clock)))

    def _publish_state_unchecked(self) -> None:
        self._publish_clock()
        frame, _tick = self._core.snapshot()
        if frame is None:
            return
        session_id, generation, tick, observed_s = self._core.observation_source()
        # Every message of one observation carries that observation's source stamp (its simulated
        # time, the /clock value it advanced), never the publish time: old state stays old.
        stamp = ros_time(observed_s)
        js = JointState()
        js.header.stamp = stamp
        for group, spec in self._groups.items():
            joints = frame.joints.get(group)
            if joints is not None:
                js.name.extend(spec["joints"])
                js.position.extend(_floats(joints)[: len(spec["joints"])])
        for group, spec in self._groups.items():
            if spec.get("gripper") and group in frame.grippers:
                js.name.append(spec["gripper"])
                js.position.append(float(frame.grippers[group]))
        self._joint_pub.publish(js)
        for group, pub in self._ee_pubs.items():
            ee = self._core.tcp_pose(frame, group)
            if ee is None:
                continue
            x, y, z, qw, qx, qy, qz = ee[:7]
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self._frame
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
            msg.pose.orientation.w, msg.pose.orientation.x = qw, qx
            msg.pose.orientation.y, msg.pose.orientation.z = qy, qz
            pub.publish(msg)
            if group == self._default_group:
                self._default_ee_pub.publish(msg)
        if not self._image_pubs:
            return
        source = (session_id, generation, tick)
        if source != self._last_image_source:
            # A new observation (a new tick, or the first of a new session even at the same tick).
            self._last_image_source = source
            self._image_msgs = {}
            for cam in self._image_pubs:
                rgb = frame.images.get(cam)
                if rgb is not None and hasattr(rgb, "shape") and len(rgb.shape) == 3:
                    self._image_msgs[cam] = _image_msg(rgb, stamp, cam)
            if self._overlay_pubs:
                self._image_msgs.update(self._overlays(frame, stamp))
        elif time.monotonic() - self._last_image_sent < IMAGE_REPUBLISH_S:
            return
        self._last_image_sent = time.monotonic()
        for key, msg in self._image_msgs.items():
            pub = (
                self._overlay_pubs.get(key[len("overlay:") :]) if key.startswith("overlay:") else self._image_pubs[key]
            )
            pub.publish(msg)

    def _overlays(self, frame: ObservationFrame, stamp) -> dict[str, Image]:
        tcp = {g: self._core.tcp_pose(frame, g) for g in self._groups}
        drawn = self._adapter.overlay_images(frame, self._core.session_info(), tcp)
        return {
            f"overlay:{cam}": _image_msg(rgb, stamp, cam)
            for cam, rgb in (drawn or {}).items()
            if cam in self._overlay_pubs and rgb is not None
        }

    # ------------------------------------------------------------------ status / mode / stop

    def _on_get_status(self, _request, response):
        response.status = self._state.to_msg(self._sim_now())
        return response

    def _on_set_mode(self, request, response):
        target = str(request.mode)
        if self._state.stop_latched and target != IDLE_MODE:
            response.success = False
            response.message = f"stop latched ({self._state.stop_policy}); request {IDLE_MODE!r} to clear it"
            response.valid_transitions = [IDLE_MODE]
            return response
        if target == self._modes.mode:
            if target == IDLE_MODE and self._state.stop_latched:
                self._state.clear_stop()
            response.success, response.message = True, f"mode {target!r} active"
            response.valid_transitions = sorted(self._modes.valid_transitions())
            return response
        decision = self._modes.can_switch(target)
        if not decision.allowed:
            response.success, response.message = False, decision.reason
            response.valid_transitions = sorted(self._modes.valid_transitions())
            return response
        if target == IDLE_MODE:
            self._core.stop("mode switched to idle")
        elif self._modes.spec(target).allows_trajectory:
            self._core.resume()
        self._modes.commit(target)
        self._state.set_active_controllers(self._modes.spec().controllers)
        if target == IDLE_MODE:
            self._state.clear_stop()
        response.success, response.message = True, f"mode {target!r} active"
        response.valid_transitions = sorted(self._modes.valid_transitions())
        self._publish_status()
        return response

    def _on_stop(self, request, response):
        policy = str(request.policy or STOP_POLICIES[0])
        if policy not in STOP_POLICIES:
            response.success = False
            response.message = f"unknown stop policy {policy!r}"
            response.cancel_latency_s = response.idle_latency_s = response.torque_off_latency_s = -1.0
            return response
        t0 = time.monotonic()
        self._state.engage_stop(policy)
        cancelled = self._core.stop(f"stop ({policy})")
        response.cancel_latency_s = time.monotonic() - t0
        self._modes.commit(IDLE_MODE)
        self._state.set_active_controllers(self._modes.spec(IDLE_MODE).controllers)
        response.idle_latency_s = time.monotonic() - t0
        # A simulated controller cannot release torque; TORQUE_OFF holds like HOLD.
        response.torque_off_latency_s = response.idle_latency_s
        response.success = True
        response.message = f"stop ({policy}) engaged; {cancelled} command(s) cancelled"
        self._publish_status()
        return response

    def _on_abort_session(self, _request, response):
        """The caller gives up on this session (e.g. its observations never arrived): stop, no result."""
        unknown = self._core.interrupt(None, "session aborted by the caller")
        response.success = True
        response.message = f"session aborted; {unknown} command(s) with unknown outcome"
        return response

    def _on_finish_session(self, _request, response):
        refusal = self._motion_refusal()
        if refusal:
            response.success, response.message = False, f"not finishing: {refusal}"
            return response
        response.success = self._core.finish()
        response.message = (
            "holding until the platform ends the session" if response.success else "no running simulation session"
        )
        return response

    def _on_cancel_command(self, request, response):
        cancelled, result = self._core.cancel_by_id(str(request.command_id), "cancelled through the task")
        response.success = cancelled
        response.outcome = str(result.get("outcome", ""))
        response.message = str(result.get("message", ""))
        response.ticks_executed = int(result.get("ticks_executed", result.get("ticks", 0)) or 0)
        self.get_logger().info(
            f"cancel {request.command_id!r}: {'cancelled' if cancelled else 'not running'} "
            f"({response.outcome}, {response.ticks_executed} ticks)"
        )
        return response

    # ------------------------------------------------------------------ motion

    def _on_move_to_pose(self, request, response):
        group, error = self._resolve_group(str(request.target_group))
        refusal = error or self._motion_refusal()
        if refusal:
            response.success, response.outcome, response.message = False, OUTCOME_REJECTED, refusal
            return response
        p, q = request.target_pose.position, request.target_pose.orientation
        cmd = Command(
            kind=COMMAND_POSE,
            group=group,
            pose=[p.x, p.y, p.z, q.w, q.x, q.y, q.z],
            velocity_scaling=float(request.velocity_scaling) or 1.0,
            generation=self._core.generation,
        )
        result = self._core.submit(cmd, MOTION_TIMEOUT_S)
        response.success = bool(result.get("success"))
        response.outcome = str(result.get("outcome", ""))
        response.message = f"[{group}] {result.get('message', '')}"
        response.execution_time_s = float(result.get("sim_time_s", 0.0))
        response.position_error_m = float(result.get("position_error_m", 0.0))
        response.orientation_error_rad = float(result.get("orientation_error_rad", 0.0))
        self.get_logger().info(
            f"move_to_pose {group} -> {response.outcome} ({response.message}; "
            f"err={response.position_error_m:.4f} m, {result.get('ticks', 0)} ticks, "
            f"max dq={result.get('max_joint_delta_rad', 0.0):.2f} rad)"
        )
        return response

    def _on_move_to_joint(self, request, response):
        refusal = self._motion_refusal()
        if refusal:
            response.success, response.outcome, response.message = False, OUTCOME_REJECTED, refusal
            return response
        named = {
            str(n): float(v)
            for n, v in zip(request.target_joint_state.name, request.target_joint_state.position, strict=False)
        }
        commands = []
        for group, spec in self._groups.items():
            joints = [str(j) for j in spec["joints"]]
            if any(j in named for j in joints):
                missing = [j for j in joints if j not in named]
                if missing:
                    response.success, response.outcome = False, OUTCOME_REJECTED
                    response.message = f"[{group}] partial arm target; missing {missing}"
                    return response
                commands.append(
                    (
                        group,
                        Command(
                            kind=COMMAND_JOINT,
                            group=group,
                            joints=[named[j] for j in joints],
                            velocity_scaling=float(request.velocity_scaling) or 1.0,
                            generation=self._core.generation,
                        ),
                    )
                )
            if spec.get("gripper") and spec["gripper"] in named:
                opening = min(1.0, max(0.0, named[spec["gripper"]]))
                commands.append(
                    (
                        group,
                        Command(kind=COMMAND_GRIPPER, group=group, opening=opening, generation=self._core.generation),
                    )
                )
        if not commands:
            response.success, response.outcome = False, OUTCOME_REJECTED
            response.message = "target joint state names no declared joint"
            return response
        total = 0.0
        for group, cmd in commands:
            result = self._core.submit(cmd, MOTION_TIMEOUT_S)
            total += float(result.get("sim_time_s", 0.0))
            if not result.get("success"):
                response.success, response.outcome = False, str(result.get("outcome", ""))
                response.message = f"[{group}] {result.get('message', '')}"
                response.execution_time_s = total
                return response
        response.success, response.outcome, response.message = True, OUTCOME_SUCCEEDED, "reached"
        response.execution_time_s = total
        return response

    def _on_compute_ik(self, request, response):
        group, error = self._resolve_group(str(request.target_group))
        if error:
            response.success, response.code, response.message = False, "UNKNOWN_GROUP", error
            return response
        refusal = ik_request_refusal(
            self._core.descriptor,
            frame_id=str(request.target.header.frame_id),
            seed_names=list(request.seed.name),
            orientation_tolerance=float(request.orientation_tolerance),
            timeout_s=float(request.timeout),
        )
        if refusal:
            response.success, response.code, response.message = False, "INVALID_REQUEST", f"[{group}] {refusal}"
            return response
        p, q = request.target.pose.position, request.target.pose.orientation
        cmd = Command(
            kind=COMMAND_IK, group=group, pose=[p.x, p.y, p.z, q.w, q.x, q.y, q.z], generation=self._core.generation
        )
        timeout = min(float(request.timeout), MOTION_TIMEOUT_S) if request.timeout > 0 else MOTION_TIMEOUT_S
        result = self._core.submit(cmd, timeout)
        if result.get("outcome") == "UNKNOWN" or "timed out" in str(result.get("message", "")):
            response.success, response.code = False, "TIMED_OUT"
            response.message = f"[{group}] {result.get('message', '')}"
            return response
        if not result.get("success"):
            response.success, response.code = False, "NO_IK_SOLUTION"
            response.message = f"[{group}] {result.get('message', '')}"
            return response
        response.success, response.code = True, "SUCCESS"
        response.message = f"[{group}] solved by the {self._core.descriptor.platform} platform"
        response.solution.name = [str(j) for j in self._groups[group]["joints"]]
        response.solution.position = [float(v) for v in result["joints"]]
        # The platform reports success only within its declared orientation tolerance (an upper bound).
        response.orientation_error = self._core.descriptor.ik_orientation_tolerance_rad
        return response

    def _on_gripper_goal(self, group: str, request) -> GoalResponse:
        gripper = str(self._groups[group]["gripper"])
        names = list(request.trajectory.joint_names)
        if names != [gripper] or not request.trajectory.points:
            self.get_logger().warning(f"gripper trajectory for {group} rejected: joints {names}, expected [{gripper}]")
            return GoalResponse.REJECT
        if self._motion_refusal():
            self.get_logger().warning(f"gripper trajectory rejected: {self._motion_refusal()}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_gripper(self, group: str, goal_handle):
        target = float(goal_handle.request.trajectory.points[-1].positions[0])
        out = FollowJointTrajectory.Result()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            out.error_string = f"[{group}] cancelled before it started"
            return out
        cmd = Command(
            kind=COMMAND_GRIPPER,
            group=group,
            opening=min(1.0, max(0.0, target)),
            generation=self._core.generation,
        )
        stop_watch = threading.Event()

        def watch_cancel() -> None:
            # An accepted cancel withdraws this command only; ticks already taken stay taken.
            while not stop_watch.wait(0.01):
                if goal_handle.is_cancel_requested:
                    self._core.cancel(cmd, "gripper goal cancelled")
                    return

        watcher = threading.Thread(target=watch_cancel, name=f"gripper_cancel_{group}", daemon=True)
        watcher.start()
        try:
            result = self._core.submit(cmd, MOTION_TIMEOUT_S)
        finally:
            stop_watch.set()
            watcher.join(timeout=1.0)
        out.error_string = f"[{group}] {result.get('message', '')}"
        if goal_handle.is_cancel_requested and result.get("outcome") == "CANCELLED":
            goal_handle.canceled()
        elif result.get("success"):
            goal_handle.succeed()
            out.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        else:
            goal_handle.abort()
            out.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
        self.get_logger().info(f"gripper {group} -> {target:.2f}: {out.error_string}")
        return out

    def destroy_node(self):
        self._adapter.stop()
        return super().destroy_node()


def main(args=None, *, node_name: str = "direct_control_runtime"):
    rclpy.init(args=args)
    node = SimulationRuntime(node_name=node_name)
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
