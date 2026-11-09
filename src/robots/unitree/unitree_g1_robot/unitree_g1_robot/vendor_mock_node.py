"""In-repo mock of the Unitree G1 vendor stack, for tests and ``use_sim: true``.

Architecture (Issue #171 PR 2): the vendor wire is the official
``unitree_sdk2py`` — the SDK's internal DDS type system
(cyclonedds-python ``IdlStruct`` classes), which rclpy cannot register as a
ROS type support, and the runtime's vendor I/O runs on the SDK's channels
exactly as it does against a robot. The mock therefore mirrors that
architecture instead of translating to ROS types: it owns the SDK channel
factory on its own (test) domain and speaks the vendor tier directly —

- ``rt/lowstate`` and the Dex3 state topics carry real ``LowState_``/
  ``HandState_`` samples built through the adapter's all-defaults factory;
- the arm_sdk / Dex3 command topics are read with SDK subscribers, so the
  runtime's ``CommandWriter`` frames land on the real vendor path;
- the RPC tier answers the runtime's SDK clients with ``Request_``/
  ``Response_`` frames through the adapter's converters, exactly as the
  robot's services would.

It does not simulate physics: the arm_sdk path is modelled as a first-order
follower that blends each joint position toward its commanded value by the
shared weight, which is exactly the observable "streaming takes effect"
behaviour the contract layer tests. Node-level rclpy plumbing remains for
parameters, logging and lifecycle (the mock is a launchable ROS node); the
DDS work is SDK-side.

Parameters expose the failure modes the bridge design has to handle, and the
ones that model vendor behaviour rather than wiring (``publish_lowstate``,
``mode_machine``, ``reject_fsm_ids``, ``silent_api_ids``, ``hand_missing``,
``action_list``, ``initial_fsm_id``) are read when they act, so a test or an
operator can set them at runtime.

Response payload shapes follow the projection's own parameter builders
(:func:`projection.fsm_id_parameter` for GetFsmId's ``{"data": id}``); the
vendor error codes below are mock-chosen stand-ins, since the vendor's own
code table is unverified (gate 6.1).
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from unitree_g1_adapter import vendor
from unitree_g1_adapter.runtime import UnitreeSdk

from unitree_g1_robot import projection
from unitree_g1_robot.runtime_node import vendor_dds_binding
from unitree_g1_robot.vendor_gateway import (
    press_sensor_state,
    rpc_request_to_dict,
    rpc_response_from_dict,
)


def _all_defaults_lowstate() -> Any:
    """A vendor ``LowState_`` sample from the adapter's all-defaults factory."""
    return vendor.lowstate_defaults()


def _all_defaults_handstate() -> Any:
    """A vendor ``HandState_`` sample from the adapter's all-defaults factory."""
    return vendor.handstate_defaults()


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
        self.declare_parameter("arm_service_status", 1)
        self.declare_parameter("motion_service_status", 0)

        topics = self._vendor["topics"]
        self._mode_pr = int(self._vendor["mode_pr"])
        self._press_sensors = int(self._vendor["dex3"]["press_sensors_per_hand"])

        self._lock = threading.Lock()
        self._tick = 0
        self._positions = [0.0] * projection.LOW_CMD_MOTOR_COUNT
        self._hand_targets = {side: [0.0] * projection.DEX3_MOTORS_PER_HAND for side in ("left", "right")}
        self._fsm_id = int(self.get_parameter("initial_fsm_id").value)
        self._hand_cmd_warned = False
        self._last_frame: dict[str, Any] | None = None
        # Request records, for tests that assert what the vendor was told.
        self.last_velocity: dict[str, Any] | None = None
        self.last_action_id: int | None = None
        self.last_tts: dict[str, Any] | None = None
        self.last_led: dict[str, Any] | None = None

        # --- vendor tier over the official SDK -------------------------------
        # rclpy cannot register the SDK's IdlStruct types as ROS type support,
        # and the runtime's vendor I/O runs on the SDK's channels exactly as it
        # does against a robot — so the mock speaks that tier through the
        # adapter's UnitreeSdk facade on its own (test) domain.
        self.declare_parameter("vendor_domain_id", 0)
        self.declare_parameter("vendor_network_interface", "")
        # The same pair rule the runtime enforces: this mock must never become
        # a fake vendor on the robot's production domain, and it must share the
        # runtime's isolated loopback domain in simulation.
        vendor_domain, vendor_network_interface = vendor_dds_binding(
            int(self.get_parameter("vendor_domain_id").value),
            str(self.get_parameter("vendor_network_interface").value or ""),
        )
        # Cross-process test aid: when set, every recorded request payload is
        # appended to this JSON-Lines file so a harness running the mock as a
        # SUBPROCESS can read the vendor-side effects (the launch composition
        # leaves it empty and the mock writes nothing).
        self.declare_parameter("records_file", "")
        self._records_file = str(self.get_parameter("records_file").value)
        self._record_seq = 0
        if self._records_file:
            Path(self._records_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self._records_file).write_text("", encoding="utf-8")
        self._sdk = UnitreeSdk(now=time.monotonic)
        self._sdk.initialize(vendor_domain, vendor_network_interface)
        self._lowstate_writer = self._sdk.command_writer(topics["low_state"], "low_state")
        self._hand_state_writers = {
            side: self._sdk.command_writer(topics[f"dex3_{side}_state"], "hand_state") for side in ("left", "right")
        }
        self._arm_sdk_reader = self._sdk.state_feed(topics["arm_sdk"], "low_cmd", handler=self._on_arm_sdk, queue_len=1)
        self._hand_cmd_readers = {
            side: self._sdk.state_feed(
                topics[f"dex3_{side}_cmd"],
                "hand_cmd",
                handler=(lambda sample, side=side: self._on_hand_cmd(side, sample)),
                queue_len=1,
            )
            for side in ("left", "right")
        }
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
            projection.AUDIO_API_IDS["get_volume"]: self._handle_get_volume,
        }
        self._rpc_response_writers = {
            key: self._sdk.command_writer(topic, "rpc_response")
            for key, topic in self._vendor["rpc"]["response_topics"].items()
        }
        for key, topic in self._vendor["rpc"]["request_topics"].items():
            self._sdk.state_feed(
                topic, "rpc_request", handler=(lambda sample, key=key: self._on_request(key, sample)), queue_len=1
            )
        # The lowstate/hand stream runs on the mock's OWN thread, not an rclpy
        # timer: the vendor write is a C call into the cyclonedds writer, and
        # driving it from an executor thread couples the whole rclpy
        # executor's health to that call (a blocked writer froze executors and
        # teardown under load). A robot's vendor stack owns its publish loop;
        # the mock mirrors that. Started here, joined in close_vendor_tier.
        # A module-scoped mock subprocess is re-configured between tests via
        # the parameter services; initial_fsm_id is read ONCE here, so a
        # runtime change must resync the live FSM state.
        self.add_on_set_parameters_callback(self._on_parameter_set)
        self._publish_stop = threading.Event()
        self._publish_thread = threading.Thread(
            target=self._publish_loop, args=(float(self.get_parameter("lowstate_period_s").value),), daemon=True
        )
        self._publish_thread.start()

    def _on_parameter_set(self, parameters: list[Any]) -> Any:
        from rcl_interfaces.msg import SetParametersResult  # noqa: PLC0415 - callback return type

        # rclpy hands the on-set callback the whole atomic parameter LIST.
        for parameter in parameters:
            if parameter.name == "initial_fsm_id":
                with self._lock:
                    self._fsm_id = int(parameter.value)
        return SetParametersResult(successful=True)

    def close_vendor_tier(self) -> None:
        """Stop the publish loop and close every SDK channel, BEFORE the rclpy
        node is destroyed.

        The SDK reader threads invoke this node's handlers; once the rclpy
        node is gone those handlers would touch a destroyed node, so teardown
        must quiesce the vendor tier first (the factory itself has no
        teardown, its threads are daemons).
        """
        self._publish_stop.set()
        self._publish_thread.join(timeout=2.0)
        # Teardown must not mask the exit path.
        with contextlib.suppress(Exception):
            self._sdk.close()

    # --- streaming state ----------------------------------------------------

    def _publish_loop(self, period_s: float) -> None:
        """The vendor stack's own publish loop (50 Hz lowstate + hand states)."""
        next_tick = time.monotonic()
        while not self._publish_stop.is_set():
            self._publish_states()
            next_tick += period_s
            sleep_s = max(next_tick - time.monotonic(), 0.001)
            if self._publish_stop.wait(sleep_s):
                return

    def _publish_states(self) -> None:
        if bool(self.get_parameter("publish_lowstate").value):
            self._lowstate_writer.write(self._build_lowstate())
        # A missing hand is a vendor-side absence: only that side goes quiet.
        missing = str(self.get_parameter("hand_missing").value)
        for side in ("left", "right"):
            if missing != "both" and side != missing:
                self._hand_state_writers[side].write(self._build_hand_state(side))

    def _build_lowstate(self) -> Any:
        message = _all_defaults_lowstate()
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

    def _build_hand_state(self, side: str) -> Any:
        message = _all_defaults_handstate()
        with self._lock:
            targets = list(self._hand_targets[side])
        # The vendor's default HandState_ already carries the 7 Dex3 slots:
        # fill them in place instead of appending (a doubled sequence would
        # read as a foreign generation downstream).
        for index, slot in enumerate(message.motor_state):
            slot.q = targets[index] if index < len(targets) else 0.0
        for _ in range(self._press_sensors):
            message.press_sensor_state.append(press_sensor_state())
        return message

    def _on_arm_sdk(self, message: Any) -> None:
        # First-order handover model: each frame's position moves toward its
        # commanded value by the shared weight, so weight 1 converges in one
        # message and weight 0 leaves the internal position untouched.
        weight = float(message.motor_cmd[projection.ARM_SDK_WEIGHT_INDEX].q)
        with self._lock:
            for name in projection.ARM_SDK_FRAME:
                index = projection.LOW_STATE_JOINT_INDEX[name]
                target = float(message.motor_cmd[index].q)
                self._positions[index] = (1.0 - weight) * self._positions[index] + weight * target
        # Frame readout for cross-process harnesses: record on CHANGE (the
        # 50 Hz tick repeats identical frames; recording each would balloon
        # the file). The weight ramp and fresh channel targets both change
        # the frame, so a streaming test sees its values land here.
        frame = {
            "weight": weight,
            "targets": {
                name: float(message.motor_cmd[index].q)
                for name, index in projection.LOW_STATE_JOINT_INDEX.items()
                if name in projection.ARM_SDK_FRAME
            },
        }
        with self._lock:
            changed = frame != self._last_frame
            if changed:
                self._last_frame = frame
        if changed:
            self._record("arm_sdk_frame", frame)

    def _on_hand_cmd(self, side: str, message: Any) -> None:
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

    def _on_request(self, service_key: str, request: Any) -> None:
        # The runtime's SDK clients publish real unitree_api Request_ frames;
        # flatten through the adapter's converter so handlers stay dict-based.
        payload = rpc_request_to_dict(request)
        api_id = int(payload["header"]["identity"]["api_id"])
        if service_key == "loco":
            # Count receipt even when fault injection intentionally drops the response.
            self._record("loco_request", api_id)
        if api_id in self._api_id_set("silent_api_ids"):
            return  # a silenced api never answers: the caller observes a timeout
        if service_key == "robot_state":
            if api_id != projection.ROBOT_STATE_API_IDS["service_list"]:
                return
            # The reported names are the robot's own, deliberately NOT derived
            # from the profile the runtime admits against: a stand-in that only
            # echoes the expectation would make the runtime's ownership check
            # prove itself. The statuses are parameters so a test or the launch
            # can report a different owner state (the preset surface needs
            # g1_arm_example stopped, the stream surface needs ai_sport running).
            self._respond(
                service_key,
                payload,
                0,
                _json(
                    [
                        {"name": "ai_sport", "status": int(self.get_parameter("motion_service_status").value)},
                        {"name": "g1_arm_example", "status": int(self.get_parameter("arm_service_status").value)},
                    ]
                ),
            )
            self._record("service_list", api_id)
            return
        handler = self._handlers.get(api_id)
        if handler is None:
            return  # unregistered api: the vendor server stays silent too
        code, data = handler(payload["parameter"])
        self._respond(service_key, payload, code, data)

    def _respond(self, service_key: str, request_payload: dict[str, Any], code: int, data: str) -> None:
        identity = request_payload["header"]["identity"]
        self._rpc_response_writers[service_key].write(
            rpc_response_from_dict(
                {
                    "header": {
                        "identity": {"id": int(identity["id"]), "api_id": int(identity["api_id"])},
                        "status": {"code": int(code)},
                    },
                    "data": data,
                }
            )
        )

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
        # The official SDK's ExecuteAction sends {"data": action_id}.
        payload = _payload(parameter)
        if payload is None or not isinstance(payload.get("data"), int):
            return MALFORMED_CODE, ""
        self._set_last("last_action_id", int(payload["data"]))
        return 0, ""

    def _handle_get_action_list(self, _parameter: str) -> tuple[int, str]:
        names = [str(name) for name in self.get_parameter("action_list").value]
        return 0, _json({"names": names})

    def _handle_get_volume(self, _parameter: str) -> tuple[int, str]:
        # The startup probe's cheap read-only call; the SDK wrapper unwraps
        # {"data": volume} to the bare number.
        return 0, _json({"data": 100})

    def _handle_tts(self, parameter: str) -> tuple[int, str]:
        # The official SDK's TtsMaker sends {"index": ..., "text": ..., "speaker_id": ...}
        # (its index counter never leaves 0 — the documented upstream quirk).
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

    def rpc_subscription_counts(self) -> dict[str, int]:
        """One RPC request subscription readiness per service (harness aid).

        The SDK tier has no rclpy-side publisher to count; the harness instead
        proves readiness by warm-probing the mock (a GetFsmId round trip), so
        this helper reports 1 once the reader exists and 0 before.
        """
        return {key: 1 for key in self._vendor["rpc"]["request_topics"]}

    def _api_id_set(self, name: str) -> set[int]:
        value = self.get_parameter(name).value
        if not value:
            return set()
        return {int(item) for item in value}

    def _set_last(self, name: str, value: Any) -> None:
        """Record one request payload; written from callbacks, read from tests."""
        self._record(name, value)
        with self._lock:
            setattr(self, name, value)

    def _record(self, name: str, value: Any) -> None:
        """Append one record to the JSON-Lines file (cross-process readout)."""
        if not self._records_file:
            return
        with self._lock:
            self._record_seq += 1
            line = json.dumps({"seq": self._record_seq, "name": name, "value": value}, separators=(",", ":"))
            with open(self._records_file, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

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
        node.close_vendor_tier()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
