"""ROS adapter for the X2 mirror session (``mirror_session`` package).

One process holds the state machine, the runtime ownership and the only
publisher of the arm and hand channels. All runtime endpoints come from the
bound runtime description by interface ID; none is configured.

Parameters (one parameter file serves the node and the release script):

``runtime_description_json``  the bound ``runtime.interface_description``
``policy_mode``               ``control_modes.model_inference.runtime_mode``
``stop_timeout_sec``          how long a stop may take (default 4.5 s): the
                              wrap-up gets it minus the 2 s release reserve;
                              the ``mirror_stop`` skill timeout must exceed it
``publish_rate_hz``           output tick (default 50 Hz)
``status_period_sec``         ``~/status`` period (default 1 s)
``publish_enabled``           false (default): the gate's output goes to the
                              preview topics only, never to the arm and hand
                              channels; the takeover and release still run.
                              Read once at startup.
``pear_frame_topic``          ``PearFrame`` / ``ActorState`` / compressed
``actor_state_topic``         camera inputs of the hooks; empty: not
``camera_topic``              subscribed
``mock_preview_topic``        both set: the window pipeline is
``mock_play_service``         ``mirror_session.mock_playback`` (the mock node's
                              animation through the gate); both empty: the
                              ``SessionHooks`` defaults
``mock_reset_pose_json``      JSON list of the 14 arm values (arm_stream
                              order) the mock window pipeline resets to,
                              approach and wrap-up included; empty: the
                              ``SessionHooks`` defaults
``mock_velocity_limits_json`` JSON list of the 14 arm speeds (rad/s,
                              arm_stream order); each joint of the mock window
                              pipeline's command moves at most this times one
                              tick (1 / ``publish_rate_hz``) from the last one
                              sent; empty: passed through
``business_hooks``            ``""``: the window pipeline above; ``demo``: the
                              demo start/cancel rules around it;
                              ``pkg.mod:Class``: a business hooks class, not
                              with the mock (``mirror_session.business``)
``business_config``           absolute path of the business config file
                              (session / timing / cover_judge / operator);
                              empty: every section empty
``hand_follow_topic``         the retarget node's 20 hand values; set: every
                              command the gate sends gets its hand half from
                              ``mirror_session.hand_overlay`` (follow while
                              mirroring, neutral otherwise); empty: unchanged
``hand_state_topic`` /        vendor hand feedback and MC's hand command, by
``hand_command_topic``        topic and type name (``*_type``), for the
                              overlay's start, unjam and staleness; empty:
                              not subscribed

Interfaces:

* ``~/set_session`` (``std_srvs/SetBool``: true = start, false = stop): called
  by the Gateway's ``mirror_session`` executor and by the operator node (false
  only);
* ``~/operator_touch`` (``std_srvs/Trigger``): STANDBY cooldown/ready ->
  MIRRORING, MIRRORING running -> cancel, nothing elsewhere; ``success`` says
  whether it did something. Serialised with ``~/set_session``;
* ``/mirror_session/state`` (``std_msgs/String``, JSON; reliable +
  transient_local, depth 1): one message per transition and a heartbeat every
  second (``StateReporter``);
* ``~/status`` (``std_msgs/String``, JSON snapshot, every
  ``status_period_sec``), with ``latency_s`` (node clock minus the commands'
  ``source_stamp_s``) and ``business_hooks``;
* ``~/arm_command_preview`` / ``~/hand_command_preview``
  (``Float64MultiArray``, every command the gate sends, whether or not
  ``publish_enabled`` is set).

In both the state and the status, ``cancel_reason`` is the reason of the most
recent cancel: a later stop or round does not clear it. ``short_rounds`` is
cleared on entering WAIT, so the transition into WAIT already reports 0.

A business hooks class is constructed in ``__init__``: a slow constructor
delays the startup.

Exits back to idle, all through ``runtime.release_to_idle``:

* normal exit (SIGINT, SIGTERM, an exception reaching ``main``): ``main``'s
  ``finally`` closes the gate and releases, spinning the executor itself;
* hard crash: launch runs ``mirror_session_release`` on process exit; it only
  calls ``set_mode``, never the arm or hand channels. Launch does not start
  it while launch itself is shutting down;
* restart: the startup check releases a leftover ``policy_stream`` before the
  session accepts ``start``.

Every release needs the runtime to answer. When the whole launch is
interrupted the runtime may exit first; the node then logs that the runtime
status is unavailable and requests nothing. Send ``mirror_stop`` before
interrupting the launch.
"""

from __future__ import annotations

import dataclasses
import json
import math
import signal
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger

from embodied_common.mirror_session_contract import NODE_NAME, STATE_PERIOD_SEC, STATE_TOPIC
from ibrobot_msgs.msg import ActorState, PearFrame, RuntimeStatus
from ibrobot_msgs.srv import GetRuntimeStatus, SetRuntimeMode
from manipulation_execution.mirror_session.business import build_hooks, load_business_config
from manipulation_execution.mirror_session.core import DEFAULT_STOP_TIMEOUT_S, MirrorSession, SessionConfig
from manipulation_execution.mirror_session.core import Phase, SessionState
from manipulation_execution.mirror_session.endpoints import ARM_JOINT_COUNT, RELEASE_INTERFACES, resolve_endpoints
from manipulation_execution.mirror_session.hand_overlay import HandOverlay
from manipulation_execution.mirror_session.hooks import HookContext
from manipulation_execution.mirror_session.inputs import InputCollector
from manipulation_execution.mirror_session.mock_playback import MockPlaybackHooks
from manipulation_execution.mirror_session.runtime import RuntimeObservation, release_to_idle


def observation_from_status(status: RuntimeStatus, now_ns: int) -> RuntimeObservation:
    try:
        description = json.loads(status.interface_description_json)
    except (TypeError, ValueError):
        description = None
    robot = description.get("robot") if isinstance(description, dict) else None
    age = None
    if 0 <= status.stamp.nanosec < 1_000_000_000:
        age = (now_ns - (status.stamp.sec * 1_000_000_000 + status.stamp.nanosec)) / 1e9
    return RuntimeObservation(
        lifecycle=status.lifecycle,
        active_mode=status.active_mode,
        stop_latched=bool(status.stop_latched),
        faults=tuple(status.faults),
        runtime_name=status.runtime_name,
        runtime_version=status.runtime_version,
        robot=robot if isinstance(robot, dict) else None,
        stamp_age_s=age,
    )


class RosRuntimePort:
    """Blocking ``get_status`` / ``set_mode``.

    While an executor thread spins, a call waits on its future. On the exit
    paths nothing spins, so ``spin_once`` is set and the call drives it.
    """

    def __init__(self, node: Node, endpoints, callback_group=None):
        self._node = node
        self._get = node.create_client(
            GetRuntimeStatus, endpoints["runtime.get_status"].endpoint, callback_group=callback_group
        )
        self._set = node.create_client(
            SetRuntimeMode, endpoints["runtime.set_mode"].endpoint, callback_group=callback_group
        )
        self.spin_once = None

    def get_status(self, timeout_s: float) -> RuntimeObservation | None:
        response = self._call(self._get, GetRuntimeStatus.Request(), timeout_s)
        if response is None:
            return None
        return observation_from_status(response.status, self._node.get_clock().now().nanoseconds)

    def set_mode(self, mode: str, timeout_s: float) -> tuple[bool, str]:
        response = self._call(self._set, SetRuntimeMode.Request(mode=mode), timeout_s)
        if response is None:
            return False, "runtime set_mode unavailable or timed out"
        return bool(response.success), response.message

    def _wait(self, done, deadline: float) -> bool:
        while not done() and time.monotonic() < deadline:
            if self.spin_once is not None:
                self.spin_once(timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
            else:
                time.sleep(0.01)
        return done()

    def _call(self, client, request, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        if not self._wait(client.service_is_ready, deadline):
            return None
        future = client.call_async(request)
        if self.spin_once is None:
            event = threading.Event()
            future.add_done_callback(lambda _future: event.set())
            event.wait(max(0.0, deadline - time.monotonic()))
        else:
            self._wait(future.done, deadline)
        if not future.done():
            client.remove_pending_request(future)
            return None
        try:
            return future.result()
        except Exception:
            return None


# Every parameter the release script and the session node declare, with its
# default. Launch may pass no other name and no other type: rclpy drops an
# override it has no declaration for without a warning, and refuses one of the
# wrong type at startup.
RELEASE_PARAMETERS = {
    "runtime_description_json": "",
    "policy_mode": "",
    "stop_timeout_sec": DEFAULT_STOP_TIMEOUT_S,
}
SESSION_PARAMETERS = {
    **RELEASE_PARAMETERS,
    "publish_rate_hz": 50.0,
    "status_period_sec": 1.0,
    "publish_enabled": False,
    "pear_frame_topic": "",
    "actor_state_topic": "",
    "camera_topic": "",
    "mock_preview_topic": "",
    "mock_play_service": "",
    "mock_reset_pose_json": "",
    "mock_velocity_limits_json": "",
    "business_hooks": "",
    "business_config": "",
    "hand_follow_topic": "",
    "hand_state_topic": "",
    "hand_state_type": "aimdk_msgs/msg/HandStateArray",
    "hand_command_topic": "",
    "hand_command_type": "aimdk_msgs/msg/HandCommandArray",
}


def _parameters(node: Node, declared=RELEASE_PARAMETERS) -> tuple[dict, str, float]:
    for name, default in declared.items():
        node.declare_parameter(name, default)
    try:
        description = json.loads(node.get_parameter("runtime_description_json").value)
    except ValueError as exc:
        raise ValueError("runtime_description_json must be the bound runtime interface description") from exc
    policy_mode = str(node.get_parameter("policy_mode").value)
    stop_timeout = float(node.get_parameter("stop_timeout_sec").value)
    return description, policy_mode, stop_timeout


def _optional_topic(node: Node, name: str) -> str:
    """An absolute ROS name, or empty for "not wired"."""
    value = node.get_parameter(name).value
    if not isinstance(value, str) or value.strip() != value or (value and not value.startswith("/")):
        raise ValueError(f"{name} must be empty or an absolute ROS name")
    return value


def _string(node: Node, name: str) -> str:
    value = node.get_parameter(name).value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _max_step(velocity: object, rate_hz: float) -> list[float]:
    """Per-tick steps from ``mock_velocity_limits_json``: 14 positive finite rad/s."""
    if (
        not isinstance(velocity, list)
        or len(velocity) != ARM_JOINT_COUNT
        or not all(type(v) in (int, float) and math.isfinite(v) and v > 0.0 for v in velocity)
    ):
        raise ValueError(f"mock_velocity_limits_json must list {ARM_JOINT_COUNT} positive finite rad/s")
    return [float(v) / rate_hz for v in velocity]


TRANSITION_FIELDS = ("from", "to", "reason", "t")


class StateReporter:
    """What goes on the state topic after a tick. No ROS.

    One message per transition, built from the transition record; with no
    transition, a heartbeat from the snapshot once ``period`` has passed since
    the last message (and at the first call). ``seq`` counts every message.
    """

    def __init__(self, period: float = STATE_PERIOD_SEC):
        self._period = period
        self._seq = 0
        self._last_at: float | None = None

    def messages(self, transitions: list[dict], snapshot: dict, now: float) -> list[dict]:
        if transitions:
            sources = transitions
        elif self._last_at is None or now - self._last_at >= self._period:
            sources = [None]
        else:
            return []
        messages = []
        for record in sources:
            current = snapshot if record is None else record
            last = snapshot.get("last_transition") if record is None else record
            self._seq += 1
            messages.append(
                {
                    "seq": self._seq,
                    "stamp": now,
                    "state": current["state"],
                    "phase": current["phase"],
                    "wrap_purpose": current["wrap_purpose"],
                    "wait_reason": current["wait_reason"],
                    "holding": current["holding"],
                    "releasing": snapshot["releasing"],
                    "cancel_reason": current["cancel_reason"],
                    "short_rounds": current["short_rounds"],
                    "last_transition": None if last is None else {key: last[key] for key in TRANSITION_FIELDS},
                }
            )
        self._last_at = now
        return messages


class LatencyMeter:
    """``latency_s``: send time minus ``source_stamp_s``, counting each stamp once.

    The gate resends the last command when the pipeline has nothing new; a
    resend carries the same stamp and is not counted again. Commands without
    a stamp are not counted.
    """

    def __init__(self):
        self._stamp: float | None = None
        self.last: float | None = None
        self.max: float | None = None
        self.n = 0
        self._sum = 0.0

    def observe(self, stamp: float | None, now: float) -> None:
        if stamp is None or not math.isfinite(stamp) or stamp == self._stamp:
            return
        self._stamp = stamp
        self.last = now - stamp
        self.max = self.last if self.max is None else max(self.max, self.last)
        self.n += 1
        self._sum += self.last

    def snapshot(self) -> dict:
        return {"last": self.last, "mean": self._sum / self.n if self.n else None, "max": self.max, "n": self.n}


class MirrorSessionNode(Node):
    def __init__(self, hooks=None, cover=None):
        super().__init__(NODE_NAME)
        description, policy_mode, stop_timeout = _parameters(self, SESSION_PARAMETERS)
        rate = float(self.get_parameter("publish_rate_hz").value)
        status_period = float(self.get_parameter("status_period_sec").value)
        if not rate > 0.0 or not status_period > 0.0:
            raise ValueError("publish_rate_hz and status_period_sec must be positive")
        self._publish_enabled = self.get_parameter("publish_enabled").value
        if type(self._publish_enabled) is not bool:
            raise ValueError("publish_enabled must be a boolean")
        input_topics = {
            name: _optional_topic(self, name) for name in ("pear_frame_topic", "actor_state_topic", "camera_topic")
        }
        mock_preview = _optional_topic(self, "mock_preview_topic")
        mock_play = _optional_topic(self, "mock_play_service")
        if bool(mock_preview) != bool(mock_play):
            raise ValueError("mock_preview_topic and mock_play_service are set together or not at all")
        if mock_preview and hooks is not None:
            raise ValueError("mock_preview_topic selects the mock window pipeline; do not pass hooks as well")
        self._business_hooks = _string(self, "business_hooks")
        business_config = _string(self, "business_config")
        reset_pose_json = _string(self, "mock_reset_pose_json")
        if reset_pose_json and not mock_preview:
            raise ValueError("mock_reset_pose_json belongs to the mock window pipeline (mock_preview_topic)")
        reset_pose = json.loads(reset_pose_json) if reset_pose_json else None
        velocity_json = _string(self, "mock_velocity_limits_json")
        if velocity_json and not mock_preview:
            raise ValueError("mock_velocity_limits_json belongs to the mock window pipeline (mock_preview_topic)")
        max_step = _max_step(json.loads(velocity_json), rate) if velocity_json else None
        self._mock = MockPlaybackHooks(self._request_play, reset_pose, max_step) if mock_preview else None
        endpoints = resolve_endpoints(description, policy_mode=policy_mode)
        business = load_business_config(business_config)
        config = SessionConfig(policy_mode, endpoints.identity, stop_timeout_s=stop_timeout, **business.timing)
        if hooks is None:
            context = HookContext(
                arm_joint_names=endpoints["joint.arm_stream"].joint_names,
                hand_joint_names=endpoints["joint.hand_stream"].joint_names,
                tick_hz=rate,
                wrap_up_budget_s=config.wrap_up_budget_s,
                publish_enabled=self._publish_enabled,
                logger=self.get_logger(),
            )
            hooks = build_hooks(self._business_hooks, business, context, self._mock)

        from robot_config.contract_utils import qos_profile_from_dict

        runtime_group = ReentrantCallbackGroup()
        self._port = RosRuntimePort(self, endpoints, runtime_group)
        arm, hand, joint_state = endpoints["joint.arm_stream"], endpoints["joint.hand_stream"], endpoints["joint.state"]
        self._arm_joints = arm.joint_names
        self._arm_publisher = self._hand_publisher = None
        if self._publish_enabled:
            self._arm_publisher = self.create_publisher(Float64MultiArray, arm.endpoint, qos_profile_from_dict(arm.qos))
            self._hand_publisher = self.create_publisher(
                Float64MultiArray, hand.endpoint, qos_profile_from_dict(hand.qos)
            )
        self._arm_preview = self.create_publisher(Float64MultiArray, "~/arm_command_preview", 10)
        self._hand_preview = self.create_publisher(Float64MultiArray, "~/hand_command_preview", 10)
        self._latency = LatencyMeter()
        self._hands = self._hand_overlay(rate)
        self._hands_last: tuple[float, ...] | None = None
        self.session = MirrorSession(config, self._port, self._send, hooks=hooks, cover=cover)

        self._inputs = InputCollector()
        inputs_group = MutuallyExclusiveCallbackGroup()
        for name, message_type, callback in (
            ("pear_frame_topic", PearFrame, self._inputs.pear),
            ("actor_state_topic", ActorState, self._inputs.actor),
            ("camera_topic", CompressedImage, self._inputs.camera),
        ):
            if input_topics[name]:
                self.create_subscription(
                    message_type, input_topics[name], callback, qos_profile_sensor_data, callback_group=inputs_group
                )
        if self._mock is not None:
            self._play_client = self.create_client(SetBool, mock_play, callback_group=ReentrantCallbackGroup())
            self.create_subscription(
                Float64MultiArray,
                mock_preview,
                lambda msg: self._mock.offer(msg.data),
                10,
                callback_group=inputs_group,
            )

        status = endpoints["runtime.status"]
        self.create_subscription(
            RuntimeStatus,
            status.endpoint,
            self._on_runtime_status,
            qos_profile_from_dict(status.qos),
            callback_group=runtime_group,
        )
        self.create_subscription(
            JointState,
            joint_state.endpoint,
            self._on_joint_state,
            qos_profile_from_dict(joint_state.qos),
            callback_group=runtime_group,
        )
        self._status_publisher = self.create_publisher(String, "~/status", 10)
        self._state_reporter = StateReporter()
        self._state_publisher = self.create_publisher(
            String,
            STATE_TOPIC,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        # One group: a touch and a start/stop are handled one after the other.
        self._requests_group = MutuallyExclusiveCallbackGroup()
        self.create_service(SetBool, "~/set_session", self._on_set_session, callback_group=self._requests_group)
        self.create_service(Trigger, "~/operator_touch", self._on_operator_touch, callback_group=self._requests_group)
        self._tick_timer = self.create_timer(1.0 / rate, self._tick, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(status_period, self._publish_status, callback_group=MutuallyExclusiveCallbackGroup())
        self._startup_timer = self.create_timer(0.1, self._startup, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(
            f"mirror session in WAIT; mode {policy_mode!r}, stop timeout {stop_timeout:.1f} s; "
            f"window pipeline: {'mock playback of ' + mock_preview if self._mock else 'hook defaults'}; "
            f"business hooks {self._business_hooks!r} ({type(hooks).__name__}), "
            f"config {business_config or '(none)'}, start/cancel confirm "
            f"{config.start_confirm_s:.1f}/{config.cancel_confirm_s:.1f} s"
        )
        if self._publish_enabled:
            self.get_logger().warning(
                f"publish_enabled=true: commands WILL be sent to {arm.endpoint} and {hand.endpoint} while mirroring"
            )
        else:
            self.get_logger().info(
                f"publish_enabled=false: nothing is sent to {arm.endpoint} or {hand.endpoint}; "
                "commands go to ~/arm_command_preview and ~/hand_command_preview only"
            )

    def _hand_overlay(self, rate: float) -> HandOverlay | None:
        follow = _optional_topic(self, "hand_follow_topic")
        if not follow:
            return None
        overlay = HandOverlay(rate)
        group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            Float64MultiArray, follow, self._guarded(lambda msg: overlay.observe_follow(msg.data, self._clock_s())),
            10, callback_group=group,
        )
        # Vendor types by name: this package does not depend on aimdk_msgs.
        from rosidl_runtime_py.utilities import get_message

        vendor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        wired = []
        for key, observe in (("hand_state", overlay.observe_state), ("hand_command", overlay.observe_command)):
            topic = _optional_topic(self, f"{key}_topic")
            if not topic:
                continue
            try:
                message_type = get_message(_string(self, f"{key}_type"))
            except (AttributeError, ModuleNotFoundError, ValueError) as exc:
                self.get_logger().error(f"hands: {key} type not loadable ({exc!r}); {topic} not subscribed")
                continue
            self.create_subscription(
                message_type, topic,
                self._guarded(lambda msg, observe=observe: observe(
                    [h.position for h in msg.left_hands], [h.position for h in msg.right_hands], self._clock_s()
                )),
                vendor_qos, callback_group=group,
            )
            wired.append(topic)
        self.get_logger().info(f"hands: follow {follow}, feedback {wired or 'none'}; neutral outside mirroring")
        return overlay

    def _guarded(self, callback):
        """A hand input callback that logs instead of raising into the executor."""

        def run(message) -> None:
            try:
                callback(message)
            except Exception as exc:
                self.get_logger().error(f"hand input dropped: {exc!r}", throttle_duration_sec=1.0)

        return run

    def _clock_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _publish(self, command) -> None:
        # The gate is the only caller, under the session lock.
        if self._hands is not None:
            following = self.session.state == SessionState.MIRRORING and self.session.phase == Phase.RUNNING
            try:
                self._hands_last = self._hands.apply(following, self._clock_s())
            except Exception as exc:  # a hand fault never reaches the gate: the hands hold their last values
                self.get_logger().error(f"hand overlay failed: {exc!r}", throttle_duration_sec=1.0)
            if self._hands_last is not None:
                command = dataclasses.replace(command, hand=self._hands_last)
        arm, hand = Float64MultiArray(data=list(command.arm)), Float64MultiArray(data=list(command.hand))
        if self._publish_enabled:
            self._arm_publisher.publish(arm)
            self._hand_publisher.publish(hand)
        self._arm_preview.publish(arm)
        self._hand_preview.publish(hand)

    def _send(self, command) -> None:
        # The gate is the only caller, under the session lock.
        self._publish(command)
        self._latency.observe(command.source_stamp_s, self.get_clock().now().nanoseconds / 1e9)

    def _request_play(self, playing: bool, round_id: int) -> None:
        # Called under the session lock: never wait here.
        if not self._play_client.service_is_ready():
            self.get_logger().warning(
                f"mock play service not ready; round {round_id} takes no preview and the gate holds"
            )
            return
        future = self._play_client.call_async(SetBool.Request(data=playing))
        future.add_done_callback(lambda done: self._on_play_response(done, playing, round_id))

    def _on_play_response(self, future, playing: bool, round_id: int) -> None:
        try:
            response = future.result()
        except Exception as exc:
            error = repr(exc)
        else:
            error = "" if response is not None and response.success else "refused"
        self._mock.acknowledge(round_id, playing, not error)
        if error:
            self.get_logger().warning(f"mock play({playing}) for round {round_id} failed: {error}")

    def _startup(self) -> None:
        self._startup_timer.cancel()
        outcome = self.session.startup()
        self.get_logger().info(f"startup check: {outcome.message}")

    def _on_runtime_status(self, msg: RuntimeStatus) -> None:
        self.session.observe_status(observation_from_status(msg, self.get_clock().now().nanoseconds))

    def _on_joint_state(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position, strict=False))
        if all(name in positions for name in self._arm_joints):
            self.session.observe_arm_position([positions[name] for name in self._arm_joints])

    def _tick(self) -> None:
        self.session.tick(dataclasses.replace(self._inputs.take(), publish_enabled=self._publish_enabled))
        transitions = self.session.drain_transitions()
        for record in transitions:
            self.get_logger().info(
                f"{record['from']} -> {record['to']} ({record['reason']}); cancel_reason={record['cancel_reason']!r} "
                f"short_rounds={record['short_rounds']}"
            )
        now = self.get_clock().now().nanoseconds / 1e9
        for message in self._state_reporter.messages(transitions, self.session.snapshot(), now):
            self._state_publisher.publish(String(data=json.dumps(message, sort_keys=True)))

    def _on_set_session(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        command = "start" if request.data else "stop"
        success, message = self.session.request(command)
        response.success = bool(success)
        response.message = f"{message} [state={self.session.state.value}]"
        if request.data:
            self.get_logger().info(f"set_session {command}: success={success} {response.message}")
        else:
            gap = self.session.stop_final_gap_deg
            gap_text = "n/a" if gap is None else f"{gap:.2f} deg"
            self.get_logger().info(
                f"set_session {command}: success={success} {response.message}; "
                f"last command before the release was {gap_text} from the reset pose"
            )
        return response

    def _on_operator_touch(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        success, message = self.session.operator_touch()
        response.success = bool(success)
        response.message = f"{message} [state={self.session.state.value}]"
        self.get_logger().info(f"operator_touch: success={success} {response.message}")
        return response

    def _publish_status(self) -> None:
        status = self.session.snapshot()
        status["publish_enabled"] = self._publish_enabled
        status["inputs"] = self._inputs.snapshot()
        status["latency_s"] = self._latency.snapshot()
        status["business_hooks"] = self._business_hooks
        if self._mock is not None:
            status["mock_playback"] = self._mock.snapshot()
        if self._hands is not None:
            status["hands"] = self._hands.snapshot(self._clock_s())
        self._status_publisher.publish(String(data=json.dumps(status, sort_keys=True)))

    def release_on_exit(self, spin_once) -> None:
        """Normal exit: nothing spins any more, so the release drives ``spin_once``."""
        self._tick_timer.cancel()
        self._port.spin_once = spin_once
        outcome = self.session.shutdown()
        # rclpy pins the severity of each call site; keep one call per level.
        if outcome.idle:
            self.get_logger().info(f"exit release: {outcome.message}")
        else:
            self.get_logger().warning(f"exit release: {outcome.message}")


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main(args=None) -> None:
    # Keep the context alive past SIGINT/SIGTERM so the exit release can still
    # reach the runtime (as robot_teleop's teleop_node does).
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _interrupt)
    executor = MultiThreadedExecutor()
    node = None
    try:
        node = MirrorSessionNode()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.release_on_exit(executor.spin_once)
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()


def release_main(args=None) -> int:
    """One-shot ``release_to_idle`` for launch's process-exit event, and nothing else."""
    rclpy.init(args=args)
    node = Node("mirror_session_release")
    executor = SingleThreadedExecutor()
    try:
        description, policy_mode, stop_timeout = _parameters(node)
        endpoints = resolve_endpoints(description, ids=RELEASE_INTERFACES, policy_mode=policy_mode)
        executor.add_node(node)
        port = RosRuntimePort(node, endpoints)
        port.spin_once = executor.spin_once
        outcome = release_to_idle(port, policy_mode, stop_timeout)
        if outcome.idle:
            node.get_logger().info(f"mirror session exit release: {outcome.message}")
        else:
            node.get_logger().error(f"mirror session exit release: {outcome.message}")
        return 0 if outcome.idle else 1
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
