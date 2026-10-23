"""In-repo mock of the Unitree G1 vendor stack, for tests and ``simulated: true``.

It speaks the real vendored IDL (``unitree_hg`` / ``unitree_api``) on the topic
names and with the QoS the profile declares — resolved through
:func:`vendor_gateway.qos_profile`, never restated here — so the bridge is
exercised against the same wire a robot uses. It does not simulate physics:
the arm_sdk path is modelled as a first-order follower that blends each joint
position toward its commanded value by the shared weight, which is exactly the
observable "streaming takes effect" behaviour the contract layer tests.

Parameters expose the failure modes the bridge design has to handle, and the
ones that model vendor behaviour rather than wiring (``publish_lowstate``,
``mode_machine``, ``reject_fsm_ids``, ``silent_api_ids``, ``hand_missing``,
``action_list``, ``initial_fsm_id``) are read when they act, so a test or an
operator can set them at runtime.

Response payload shapes follow the projection's own parameter builders
(:func:`projection.fsm_id_parameter` for GetFsmId's ``{"data": id}``); the
vendor error codes below are mock-chosen stand-ins, since the vendor's own code
table is unverified (gate 6.1).
"""

from __future__ import annotations

import json
import threading
from functools import partial
from pathlib import Path
from typing import Any

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from unitree_g1_robot import projection
from unitree_g1_robot.vendor_gateway import (
    HandCmd,
    HandState,
    LowCmd,
    LowState,
    MotorState,
    PressSensorState,
    Request,
    Response,
    qos_profile,
)

#: Vendor error code the mock answers a rejected SetFsmId with (0 is success).
REJECT_CODE = 1
#: Vendor error code for a request parameter that does not parse as expected.
MALFORMED_CODE = 2


def _default_profile() -> Path:
    """The installed profile, as the console_script entry point finds it.

    The share directory (setup.py installs ``profiles/*.yaml`` there) resolves
    under both ``colcon build --symlink-install`` and a plain install; the
    source tree's own path does not, because a symlinked egg points back into
    ``build/`` where no profiles live.
    """
    return Path(get_package_share_directory("unitree_g1_robot")) / "profiles" / "g1_29dof_dex3.yaml"


def _json(payload: Any) -> str:
    """Serialize one response payload the way the projection's builders do."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _payload(parameter: str) -> dict[str, Any] | None:
    """Parse a request parameter, or None when it is not the expected JSON object."""
    try:
        payload = json.loads(parameter)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


class UnitreeVendorMock(Node):
    """Minimum G1 vendor surface: lowstate/arm_sdk/hand streams plus the RPC tier."""

    def __init__(self, **node_kwargs: Any) -> None:
        super().__init__("unitree_vendor_mock", **node_kwargs)
        self.declare_parameter("profile", str(_default_profile()))
        with open(Path(str(self.get_parameter("profile").value))) as handle:
            self._vendor: dict[str, Any] = yaml.safe_load(handle)["vendor"]

        # LowState feedback period, read once (a timer cannot be re-ticked at
        # runtime); every other parameter below is read where it acts.
        self.declare_parameter("lowstate_period_s", 0.02)
        # Fault injection: the reported generation, a dead feedback stream, a
        # hand that never publishes, FSM ids and api ids the vendor refuses or
        # ignores, and the action table GetActionList reports.
        self.declare_parameter("mode_machine", int(self._vendor["mode_machine"]))
        self.declare_parameter("publish_lowstate", True)
        self.declare_parameter("hand_missing", "")
        # rclpy infers BYTE_ARRAY from an empty list default, which then rejects
        # integer arrays set later; dynamic typing keeps these runtime-settable.
        self.declare_parameter("reject_fsm_ids", descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("silent_api_ids", descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter(
            "action_list", [str(entry["action_name"]) for entry in self._vendor["arm_actions"].values()]
        )
        self.declare_parameter("initial_fsm_id", projection.FSM_IDS["zero_torque"])

        topics = self._vendor["topics"]
        qos = self._vendor["qos"]
        self._mode_pr = int(self._vendor["mode_pr"])
        self._press_sensors = int(self._vendor["dex3"]["press_sensors_per_hand"])

        self._lock = threading.Lock()
        self._tick = 0
        self._positions = [0.0] * projection.LOW_CMD_MOTOR_COUNT
        self._hand_targets = {side: [0.0] * projection.DEX3_MOTORS_PER_HAND for side in ("left", "right")}
        self._fsm_id = int(self.get_parameter("initial_fsm_id").value)
        self._hand_cmd_warned = False
        # Request records, for tests that assert what the vendor was told.
        self.last_velocity: dict[str, Any] | None = None
        self.last_action_id: int | None = None
        self.last_tts: dict[str, Any] | None = None
        self.last_led: dict[str, Any] | None = None

        self._lowstate_pub = self.create_publisher(LowState, topics["low_state"], qos_profile(qos, "low_state"))
        self._hand_pubs = {
            side: self.create_publisher(HandState, topics[f"dex3_{side}_state"], qos_profile(qos, "dex3_state"))
            for side in ("left", "right")
        }
        self.create_subscription(LowCmd, topics["arm_sdk"], self._on_arm_sdk, qos_profile(qos, "arm_sdk"))
        for side in ("left", "right"):
            self.create_subscription(
                HandCmd, topics[f"dex3_{side}_cmd"], partial(self._on_hand_cmd, side), qos_profile(qos, "dex3_cmd")
            )

        self._handlers: dict[int, Any] = {
            projection.LOCO_API_IDS["get_fsm_id"]: self._handle_get_fsm_id,
            projection.LOCO_API_IDS["set_fsm_id"]: self._handle_set_fsm_id,
            projection.LOCO_API_IDS["set_velocity"]: self._handle_set_velocity,
            projection.LOCO_API_IDS["switch_to_user_ctrl"]: self._handle_ok,
            projection.LOCO_API_IDS["switch_to_internal_ctrl"]: self._handle_ok,
            projection.ARM_ACTION_API_IDS["execute_action"]: self._handle_execute_action,
            projection.ARM_ACTION_API_IDS["get_action_list"]: self._handle_get_action_list,
            projection.AUDIO_API_IDS["tts"]: self._handle_tts,
            projection.AUDIO_API_IDS["set_rgb_led"]: self._handle_set_led,
        }
        self._rpc_pubs: dict[str, Any] = {}
        for key, topic in self._vendor["rpc"]["response_topics"].items():
            self._rpc_pubs[key] = self.create_publisher(Response, topic, qos_profile(qos, "rpc_response"))
        for key, topic in self._vendor["rpc"]["request_topics"].items():
            response_pub = self._rpc_pubs[key]
            self.create_subscription(
                Request, topic, partial(self._on_request, response_pub), qos_profile(qos, "rpc_request")
            )

        self.create_timer(float(self.get_parameter("lowstate_period_s").value), self._publish_states)

    # --- streaming state ----------------------------------------------------

    def _publish_states(self) -> None:
        if bool(self.get_parameter("publish_lowstate").value):
            self._lowstate_pub.publish(self._build_lowstate())
        # A missing hand is a vendor-side absence: only that side goes quiet.
        missing = str(self.get_parameter("hand_missing").value)
        for side in ("left", "right"):
            if side != missing:
                self._hand_pubs[side].publish(self._build_hand_state(side))

    def _build_lowstate(self) -> LowState:
        message = LowState()
        message.mode_pr = self._mode_pr
        message.mode_machine = int(self.get_parameter("mode_machine").value)
        with self._lock:
            self._tick += 1
            tick = self._tick
            positions = list(self._positions)
        message.tick = tick
        message.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        message.imu_state.gyroscope = [0.0, 0.0, 0.0]
        message.imu_state.accelerometer = [0.0, 0.0, 9.81]
        for index in range(projection.LOW_CMD_MOTOR_COUNT):
            message.motor_state[index].q = positions[index]
        return message

    def _build_hand_state(self, side: str) -> HandState:
        message = HandState()
        with self._lock:
            targets = list(self._hand_targets[side])
        for q in targets:
            state = MotorState()
            state.q = q
            message.motor_state.append(state)
        for _ in range(self._press_sensors):
            message.press_sensor_state.append(PressSensorState())
        return message

    def _on_arm_sdk(self, message: LowCmd) -> None:
        # First-order handover model: each frame's position moves toward its
        # commanded value by the shared weight, so weight 1 converges in one
        # message and weight 0 leaves the internal position untouched.
        weight = float(message.motor_cmd[projection.ARM_SDK_WEIGHT_INDEX].q)
        with self._lock:
            for name in projection.ARM_SDK_FRAME:
                index = projection.LOW_STATE_JOINT_INDEX[name]
                target = float(message.motor_cmd[index].q)
                self._positions[index] = (1.0 - weight) * self._positions[index] + weight * target

    def _on_hand_cmd(self, side: str, message: HandCmd) -> None:
        # A HandCmd that does not carry exactly the Dex3 7 slots would make the
        # published state a hand the gateway refuses (handstate_q demands 7);
        # dropped once with a warning instead of poisoning the state stream.
        if len(message.motor_cmd) != projection.DEX3_MOTORS_PER_HAND:
            if not self._hand_cmd_warned:
                self._hand_cmd_warned = True
                self.get_logger().warning(
                    f"dropping {side} hand cmd with {len(message.motor_cmd)} motor slots; "
                    f"a Dex3 hand commands exactly {projection.DEX3_MOTORS_PER_HAND}"
                )
            return
        with self._lock:
            self._hand_targets[side] = [float(slot.q) for slot in message.motor_cmd]

    # --- RPC tier -----------------------------------------------------------

    def _on_request(self, response_pub: Any, request: Request) -> None:
        api_id = int(request.header.identity.api_id)
        if api_id in self._api_id_set("silent_api_ids"):
            return  # a silenced api never answers: the caller observes a timeout
        handler = self._handlers.get(api_id)
        if handler is None:
            return  # unregistered api: the vendor server stays silent too
        code, data = handler(request.parameter)
        self._respond(response_pub, request, code, data)

    def _respond(self, response_pub: Any, request: Request, code: int, data: str) -> None:
        response = Response()
        response.header.identity.id = int(request.header.identity.id)
        response.header.identity.api_id = int(request.header.identity.api_id)
        response.header.status.code = int(code)
        response.data = data
        response_pub.publish(response)

    def _handle_ok(self, _parameter: str) -> tuple[int, str]:
        return 0, ""

    def _handle_get_fsm_id(self, _parameter: str) -> tuple[int, str]:
        with self._lock:
            fsm_id = self._fsm_id
        return 0, _json(projection.fsm_id_parameter(fsm_id))

    def _handle_set_fsm_id(self, parameter: str) -> tuple[int, str]:
        payload = _payload(parameter)
        if payload is None or not isinstance(payload.get("data"), int):
            return MALFORMED_CODE, ""
        fsm_id = int(payload["data"])
        if fsm_id in self._api_id_set("reject_fsm_ids"):
            return REJECT_CODE, ""
        with self._lock:
            self._fsm_id = fsm_id
        return 0, ""

    def _handle_set_velocity(self, parameter: str) -> tuple[int, str]:
        payload = _payload(parameter)
        if payload is None:
            return MALFORMED_CODE, ""
        self._set_last("last_velocity", payload)
        return 0, ""

    def _handle_execute_action(self, parameter: str) -> tuple[int, str]:
        payload = _payload(parameter)
        if payload is None or not isinstance(payload.get("action_id"), int):
            return MALFORMED_CODE, ""
        self._set_last("last_action_id", int(payload["action_id"]))
        return 0, ""

    def _handle_get_action_list(self, _parameter: str) -> tuple[int, str]:
        names = [str(name) for name in self.get_parameter("action_list").value]
        return 0, _json({"names": names})

    def _handle_tts(self, parameter: str) -> tuple[int, str]:
        payload = _payload(parameter)
        if payload is None:
            return MALFORMED_CODE, ""
        self._set_last("last_tts", payload)
        return 0, ""

    def _handle_set_led(self, parameter: str) -> tuple[int, str]:
        payload = _payload(parameter)
        if payload is None:
            return MALFORMED_CODE, ""
        self._set_last("last_led", payload)
        return 0, ""

    # --- parameters ----------------------------------------------------------

    def _api_id_set(self, name: str) -> set[int]:
        value = self.get_parameter(name).value
        if not value:
            return set()
        return {int(item) for item in value}

    def _set_last(self, name: str, value: Any) -> None:
        """Record one request payload; written from callbacks, read from tests."""
        with self._lock:
            setattr(self, name, value)

    def record(self, name: str) -> Any:
        """One recorded request payload, read under the same lock it is written under."""
        with self._lock:
            return getattr(self, name)


def main(argv=None) -> int:
    # SIGTERM included so `timeout`/systemd stops exit spin via the ordinary
    # ExternalShutdownException path instead of dying inside an rcl wait set.
    rclpy.init(
        args=argv,
        signal_handler_options=rclpy.signals.SignalHandlerOptions.ALL,
    )
    node = UnitreeVendorMock()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
