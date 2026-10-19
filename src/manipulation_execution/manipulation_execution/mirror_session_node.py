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

Interfaces: ``~/set_session`` (``std_srvs/SetBool``: true = start, false =
stop; only the Gateway's ``mirror_session`` executor calls it), ``~/status``
(``std_msgs/String``, JSON snapshot), and ``~/arm_command_preview`` /
``~/hand_command_preview`` (``Float64MultiArray``, every command the gate sends,
whether or not ``publish_enabled`` is set).

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

import json
import signal
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool

from ibrobot_msgs.msg import ActorState, PearFrame, RuntimeStatus
from ibrobot_msgs.srv import GetRuntimeStatus, SetRuntimeMode
from manipulation_execution.mirror_session.core import DEFAULT_STOP_TIMEOUT_S, MirrorSession, SessionConfig
from manipulation_execution.mirror_session.endpoints import RELEASE_INTERFACES, resolve_endpoints
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


class MirrorSessionNode(Node):
    def __init__(self, hooks=None, cover=None):
        super().__init__("mirror_session")
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
        self._mock = MockPlaybackHooks(self._request_play) if mock_preview else None
        if self._mock is not None:
            hooks = self._mock
        endpoints = resolve_endpoints(description, policy_mode=policy_mode)
        config = SessionConfig(policy_mode, endpoints.identity, stop_timeout_s=stop_timeout)

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
        self.session = MirrorSession(config, self._port, self._publish, hooks=hooks, cover=cover)

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
        self.create_service(
            SetBool, "~/set_session", self._on_set_session, callback_group=MutuallyExclusiveCallbackGroup()
        )
        self._tick_timer = self.create_timer(1.0 / rate, self._tick, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(status_period, self._publish_status, callback_group=MutuallyExclusiveCallbackGroup())
        self._startup_timer = self.create_timer(0.1, self._startup, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(
            f"mirror session in WAIT; mode {policy_mode!r}, stop timeout {stop_timeout:.1f} s; "
            f"window pipeline: {'mock playback of ' + mock_preview if self._mock else 'hook defaults'}"
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

    def _publish(self, command) -> None:
        # The gate is the only caller, under the session lock.
        arm, hand = Float64MultiArray(data=list(command.arm)), Float64MultiArray(data=list(command.hand))
        if self._publish_enabled:
            self._arm_publisher.publish(arm)
            self._hand_publisher.publish(hand)
        self._arm_preview.publish(arm)
        self._hand_preview.publish(hand)

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
        self.session.tick(self._inputs.take())

    def _on_set_session(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        command = "start" if request.data else "stop"
        success, message = self.session.request(command)
        response.success = bool(success)
        response.message = f"{message} [state={self.session.state.value}]"
        self.get_logger().info(f"set_session {command}: success={success} {response.message}")
        return response

    def _publish_status(self) -> None:
        status = self.session.snapshot()
        status["publish_enabled"] = self._publish_enabled
        status["inputs"] = self._inputs.snapshot()
        if self._mock is not None:
            status["mock_playback"] = self._mock.snapshot()
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
