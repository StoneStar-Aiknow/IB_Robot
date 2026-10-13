"""Unitree G1 runtime: the public runtime contract served over the vendor DDS tier.

This is the wrapper layer. Generic IB-Robot packages talk only to contract
names (``/runtime_status``, ``/runtime/set_mode``, ``/runtime/get_status``,
``/runtime/stop``, ``/joint_states``, ``/imu/data`` and the profile's declared
command channels) and never link ``unitree_hg`` or ``unitree_api``; the
vendor's topics, api ids, FSM ids and joint order stay behind this node.

What the node does NOT do (design D1/D11): instantiate a controller manager,
load a ros2_control hardware component, or publish on the unprotected
``rt/lowcmd`` tier — ``expert_tier_gate`` refuses a profile that enables it.

Wiring the vendor DDS graph (the ``rt/lowstate`` / Dex3 state subscriptions,
the ``rt/arm_sdk`` / Dex3 command publishers and the ``rt/api/*`` RPC
request/response pair) is this node's own job per ``vendor_gateway``'s
contract; the vendor message classes themselves come re-exported from
``vendor_gateway``, which stays the only module that imports a vendor package,
and every value conversion goes through ``vendor_gateway`` and ``projection``.

Key behaviours, per openspec design:

* D4 shared-weight frame: entering stream latches fresh measured positions as
  hold targets for all 17 arm_sdk joints, ``SwitchToUserCtrl`` hands the frame
  over, and the shared weight ramps to 1; leaving stream ramps the weight back
  to 0 before the mode is reported inactive. A channel that goes quiet simply
  stops updating the latch — its slots keep the last commanded value, which is
  exactly the profile's ``stream_staleness_s`` policy, so expiry needs no
  timer of its own: nothing is zero-filled and no measured value is ever
  written back into a command slot.
* D5 modes and stop: locomotion is entered through ``SetFsmId(start)`` and
  confirmed by ``GetFsmId``; ``StopRuntime(HOLD)`` closes admission first,
  then zeroes velocity, freezes the latched targets and ramps the weight to 0;
  ``TORQUE_OFF`` maps to the damping FSM and is refused while the reported
  posture is not in the profile's safe set. A stop preempts a mode switch
  that is still waiting on the vendor.
* D7 capability mapping: the profile's arm action table is compared against
  ``GetActionList`` at startup; actions the vendor does not report are not
  advertised and a warning names each one.
* D10 configuration check: the first LowState's ``mode_machine`` must match
  the profile and both Dex3 state streams must be publishing, or the runtime
  faults, never reaches ACTIVE and serves no command channel.
* D6 public surface: the interface description served in RuntimeStatus is
  supplied by the launch entry, which projects it from the profile alone (no
  URDF, so no public model is published); this node provides it verbatim and
  never rebuilds it.
"""

from __future__ import annotations

import json
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import GetRuntimeStatus, SetLedPattern, SetRuntimeMode, SpeakText, StopRuntime
from robot_runtime.contract import (
    GET_STATUS_SERVICE,
    IDLE_MODE,
    LIFECYCLE_ACTIVE,
    SET_MODE_SERVICE,
    STATUS_TOPIC,
    STOP_POLICIES,
    STOP_SERVICE,
)
from robot_runtime.modes import ModeModel, ModeModelConfig
from robot_runtime.profile import load_profile
from robot_runtime.state import RuntimeState
from unitree_g1_robot import projection
from unitree_g1_robot import vendor_gateway as vg
from unitree_g1_robot.named_motion import NamedMotionServer
from unitree_g1_robot.projection import CommandRejected, ProjectionError

#: Neutral IMU endpoint, as the profile's interfaces declare it. The frame_id
#: stays unset: which URDF frame LowState.imu_state reports is unverified
#: (gate 6.1) and a guess would mislabel every consumer of the frame.
IMU_TOPIC = "/imu/data"

#: Neutral interaction endpoints; the profile's interfaces declare the same
#: ones and are read at wiring time (these are the fallbacks).
SPEAK_SERVICE = "/speech/speak"
LED_SERVICE = "/led/set_pattern"

#: Vendor RPC services the bridge talks to: locomotion and FSM switches, arm
#: actions, and the voice service behind TTS and the LED.
RPC_SERVICES_IN_USE = ("loco", "arm", "audio")

#: SetVelocity re-issue cadence check. The command itself carries the
#: profile's duration; this tick only decides when a re-issue is due, and a
#: tenth of that duration keeps the re-issue prompt without spamming the tier.
_VELOCITY_TICK_S = 0.1

#: How long startup waits for the vendor's RPC servers to match our request
#: publishers. A request published before DDS matching completes is silently
#: dropped, so without this the startup queries would observe timeouts that
#: are really a discovery race.
_RPC_MATCH_TIMEOUT_S = 2.0

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
_RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def _default_profile() -> Path:
    """The installed profile, as the console_script entry point finds it."""
    return Path(get_package_share_directory("unitree_g1_robot")) / "profiles" / "g1_29dof_dex3.yaml"


class UnitreeG1RuntimeNode(Node):
    """Bridge between the public runtime contract and the G1 vendor DDS tier."""

    def __init__(self, **node_kwargs: Any) -> None:
        super().__init__("unitree_g1_runtime", **node_kwargs)
        self.declare_parameter("profile", str(_default_profile()))
        self.declare_parameter("startup_timeout_s", 10.0)
        # The launch entry builds the description from the same profile it
        # feeds the synthetic sensor publishers, so what the runtime advertises
        # and what simulated transport produces cannot drift apart.
        self.declare_parameter("interface_description_json", "")
        self.declare_parameter("initial_mode", "")

        self._profile = load_profile(str(self.get_parameter("profile").value))
        # Only idle can be initial on the G1: stream and locomotion are entered
        # through vendor-side switches (SwitchToUserCtrl, SetFsmId), so a
        # runtime that boots "into" one would claim a control state the vendor
        # never entered.
        initial_mode = str(self.get_parameter("initial_mode").value or "")
        if initial_mode and initial_mode != IDLE_MODE:
            raise RuntimeError(
                f"unsupported initial_mode {initial_mode!r}: a G1 runtime starts in {IDLE_MODE!r}; "
                "non-idle modes are entered through the mode service"
            )
        supplied = str(self.get_parameter("interface_description_json").value or "")
        # RuntimeState validates the description; an empty parameter leaves the
        # status without one (a bare node run outside the launch entry).
        interface_description = json.loads(supplied) if supplied else None
        self._vendor: dict[str, Any] = self._profile.get("vendor") or {}
        projection.expert_tier_gate(bool((self._vendor.get("expert_joint_tier") or {}).get("enabled", False)))

        modes = ModeModel(ModeModelConfig.from_profile(self._profile))
        self._state = RuntimeState(
            name=str(self._profile["runtime"]["name"]),
            version=str(self._profile["runtime"]["version"]),
            capabilities=self._profile["capabilities"],
            modes=modes,
            on_change=self._publish_status,
            interface_description=interface_description,
        )
        self._modes = modes

        self._mode_pr = int(self._vendor["mode_pr"])
        self._mode_machine_expected = int(self._vendor["mode_machine"])
        self._joint_order = [str(joint) for joint in self._profile["joints"]]
        self._joint_limits: dict[str, Any] = self._vendor.get("joint_limits") or {}

        arm_sdk = self._vendor.get("arm_sdk") or {}
        self._arm_sdk_period_s = 1.0 / max(float(arm_sdk.get("rate_hz", 50.0)), 1.0)
        # Ramp-up hands the frame to the user; ramp-down hands it back, and
        # the rates are separate profile keys because they bound different
        # paths (stream entry vs the stop/exit latency the stop bounds declare).
        self._weight_ramp_up_rate = float(arm_sdk.get("weight_ramp_rate", projection.EXAMPLE_WEIGHT_RAMP_RATE))
        self._weight_ramp_down_rate = float(
            (self._vendor.get("stop") or {}).get("weight_ramp_down_rate", self._weight_ramp_up_rate)
        )
        self._arm_sdk_kp = float(arm_sdk.get("default_kp", projection.EXAMPLE_ARM_SDK_KP))
        self._arm_sdk_kd = float(arm_sdk.get("default_kd", projection.EXAMPLE_ARM_SDK_KD))
        # Entering stream demands feedback no older than this (design D4); the
        # same window decides which sources a /joint_states aggregate may use.
        self._feedback_window_s = float(arm_sdk.get("feedback_staleness_s", 0.5))

        dex3 = self._vendor.get("dex3") or {}
        self._dex3_period_s = 1.0 / max(float(dex3.get("rate_hz", 100.0)), 1.0)
        self._dex3_kp = float(dex3.get("default_kp", projection.EXAMPLE_DEX3_KP))
        self._dex3_kd = float(dex3.get("default_kd", projection.EXAMPLE_DEX3_KD))

        self._locomotion_limits = projection.LocomotionLimits.from_profile(self._vendor.get("locomotion"))
        self._velocity_duration_s = float((self._vendor.get("locomotion") or {}).get("command_duration_s", 1.0))
        self._twist_staleness_s = float(
            ((self._profile.get("capabilities") or {}).get("base.cmd_vel") or {}).get("staleness_s", 0.5)
        )
        self._rpc_timeout_s = float((self._vendor.get("rpc") or {}).get("timeout_s", 5.0))
        self._startup_timeout_s = float(self.get_parameter("startup_timeout_s").value)
        self._joint_state_rate_hz = float(
            ((self._profile.get("capabilities") or {}).get("joint.state") or {}).get("rate_hz", 50.0)
        )

        # --- shared mutable state (guarded by _lock) -------------------------
        self._lock = threading.RLock()
        # The 17-joint arm_sdk latch: seeded from fresh measurements at stream
        # entry, updated only by admitted channel messages after that.
        self._latch: dict[str, float] = {}
        self._hand_latch: dict[str, dict[str, float]] = {"left": {}, "right": {}}
        self._weight = 0.0
        self._weight_target = projection.MODE_WEIGHT_TARGETS[IDLE_MODE]
        self._twist: tuple[float, float, float] | None = None
        self._twist_stamp = 0.0
        self._velocity_pending: int | None = None
        self._velocity_sent: tuple[float, float, float] | None = None
        self._last_velocity_sent_s = 0.0
        self._confirmed_actions: set[str] = set()

        # --- vendor feedback (guarded by _lock) ------------------------------
        self._lowstate_positions: dict[str, float] = {}
        self._lowstate_stamp = 0.0
        self._lowstate_clock: Any = None
        self._mode_machine_observed: int | None = None
        self._imu_payload: dict[str, Any] | None = None
        self._hand_positions: dict[str, dict[str, float]] = {"left": {}, "right": {}}
        self._hand_stamps: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._hand_clocks: dict[str, Any] = {"left": None, "right": None}

        # A D10 fault is a generation mismatch: it is never cleared, and a
        # stop latch cleared on top of it must not resurrect an ACTIVE runtime.
        self._startup_fault: str | None = None
        self._stop_lock = threading.Lock()

        # --- callback groups ---------------------------------------------------
        # Vendor state feeds are high-rate and must never queue behind a
        # blocking service call; the output ticks must keep running while a
        # mode switch or a stop blocks its own callback waiting on the vendor.
        self._vendor_cb = MutuallyExclusiveCallbackGroup()
        self._tick_cb = MutuallyExclusiveCallbackGroup()
        self._state_cb = MutuallyExclusiveCallbackGroup()
        self._rpc_cb = ReentrantCallbackGroup()
        self._service_cb = MutuallyExclusiveCallbackGroup()
        self._stop_cb = ReentrantCallbackGroup()
        self._interaction_cb = MutuallyExclusiveCallbackGroup()
        self._action_cb = ReentrantCallbackGroup()

        self._status_pub = self.create_publisher(RuntimeStatus, STATUS_TOPIC, 10)
        self._joint_pub = self.create_publisher(JointState, str(self._profile["joint_state_topic"]), _RELIABLE_QOS)
        self._imu_pub = self.create_publisher(Imu, IMU_TOPIC, _SENSOR_QOS)

        self._setup_vendor_io()
        self._setup_contract_surface()

        self.create_timer(self._arm_sdk_period_s, self._arm_sdk_tick, callback_group=self._tick_cb)
        self.create_timer(self._dex3_period_s, self._dex3_tick, callback_group=self._tick_cb)
        self.create_timer(_VELOCITY_TICK_S, self._velocity_tick, callback_group=self._tick_cb)
        self.create_timer(
            1.0 / max(self._joint_state_rate_hz, 1.0), self._publish_joint_states, callback_group=self._state_cb
        )
        self.create_timer(1.0, self._publish_status, callback_group=self._state_cb)
        self._startup_thread = threading.Thread(target=self._startup_checks, daemon=True)
        self._startup_thread.start()

    # --- wiring -------------------------------------------------------------

    def _setup_vendor_io(self) -> None:
        """Subscribe to the vendor state feeds and create the command/RPC endpoints.

        The vendor message classes come re-exported from ``vendor_gateway``,
        and ``require_vendor`` has already refused to start without them, so
        this wiring is the only place vendor types are named in this module.
        """
        vg.require_vendor()
        topics = self._vendor["topics"]
        qos = self._vendor.get("qos")
        self.create_subscription(
            vg.LowState,
            topics["low_state"],
            self._on_lowstate,
            vg.qos_profile(qos, "low_state"),
            callback_group=self._vendor_cb,
        )
        for side in ("left", "right"):
            self.create_subscription(
                vg.HandState,
                topics[f"dex3_{side}_state"],
                partial(self._on_hand_state, side),
                vg.qos_profile(qos, "dex3_state"),
                callback_group=self._vendor_cb,
            )
        self._arm_sdk_pub = self.create_publisher(vg.LowCmd, topics["arm_sdk"], vg.qos_profile(qos, "arm_sdk"))
        self._hand_pubs = {
            side: self.create_publisher(vg.HandCmd, topics[f"dex3_{side}_cmd"], vg.qos_profile(qos, "dex3_cmd"))
            for side in ("left", "right")
        }
        self._rpc: dict[str, vg.RpcClient] = {}
        self._rpc_publishers: dict[str, Any] = {}
        for service in RPC_SERVICES_IN_USE:
            request_pub = self.create_publisher(
                vg.Request, self._vendor["rpc"]["request_topics"][service], vg.qos_profile(qos, "rpc_request")
            )
            client = vg.RpcClient(publish=request_pub.publish, now=time.monotonic)
            self._rpc[service] = client
            self._rpc_publishers[service] = request_pub
            self.create_subscription(
                vg.Response,
                self._vendor["rpc"]["response_topics"][service],
                client.on_response,
                vg.qos_profile(qos, "rpc_response"),
                callback_group=self._rpc_cb,
            )

    def _setup_contract_surface(self) -> None:
        self.create_service(SetRuntimeMode, SET_MODE_SERVICE, self._on_set_mode, callback_group=self._service_cb)
        self.create_service(GetRuntimeStatus, GET_STATUS_SERVICE, self._on_get_status, callback_group=self._stop_cb)
        self.create_service(StopRuntime, STOP_SERVICE, self._on_stop, callback_group=self._stop_cb)
        for channel in self._profile.get("command_channels", []):
            topic = str(channel["topic"])
            if str(channel.get("type", "float64_array")) == "twist":
                self.create_subscription(
                    Twist, topic, lambda message, channel=channel: self._on_twist(channel, message), _SENSOR_QOS
                )
            else:
                self.create_subscription(
                    Float64MultiArray,
                    topic,
                    lambda message, channel=channel: self._on_channel(channel, message),
                    _SENSOR_QOS,
                )
        if self._state.has_capability("interaction.tts"):
            interface = (self._profile.get("interfaces") or {}).get("speech.speak") or {}
            self.create_service(
                SpeakText,
                str(interface.get("endpoint", SPEAK_SERVICE)),
                self._on_speak,
                callback_group=self._interaction_cb,
            )
        if self._state.has_capability("interaction.led"):
            interface = (self._profile.get("interfaces") or {}).get("led.set_pattern") or {}
            self.create_service(
                SetLedPattern,
                str(interface.get("endpoint", LED_SERVICE)),
                self._on_led,
                callback_group=self._interaction_cb,
            )
        self._named_motion = NamedMotionServer(self)

    # --- startup (design D10) --------------------------------------------------

    def _startup_checks(self) -> None:
        """Reach ACTIVE only when the unit matches the profile (design D10)."""
        deadline = time.monotonic() + self._startup_timeout_s
        while True:
            with self._lock:
                have_lowstate = self._lowstate_stamp > 0.0
                have_hands = all(self._hand_stamps[side] > 0.0 for side in ("left", "right"))
                observed = self._mode_machine_observed
            if have_lowstate and have_hands:
                break
            if time.monotonic() >= deadline:
                missing: list[str] = []
                if not have_lowstate:
                    missing.append("the first LowState frame")
                with self._lock:
                    quiet = [side for side in ("left", "right") if self._hand_stamps[side] <= 0.0]
                missing += [f"the {side} Dex3 state stream" for side in quiet]
                self._fail_startup(
                    "vendor feedback missing at startup: " + "; ".join(missing) + " (design D10); "
                    "not entering ACTIVE and serving no command channel"
                )
                return
            time.sleep(0.05)

        fault = projection.mode_machine_fault(observed, self._mode_machine_expected)
        if fault is not None:
            self._fail_startup(fault)
            return

        for service in RPC_SERVICES_IN_USE:
            self._wait_for_rpc_peer(service)
        self._confirm_vendor_actions()
        self._state.set_lifecycle(LIFECYCLE_ACTIVE)

    def _fail_startup(self, detail: str) -> None:
        """Hold the runtime out of ACTIVE for good: a D10 fault is a generation mismatch."""
        self._startup_fault = detail
        self._state.set_status("FAULTED", [detail])

    def _wait_for_rpc_peer(self, service: str) -> None:
        """Wait until the vendor's RPC server matched our request publisher."""
        publisher = self._rpc_publishers[service]
        deadline = time.monotonic() + _RPC_MATCH_TIMEOUT_S
        while time.monotonic() < deadline and publisher.get_subscription_count() == 0:
            time.sleep(0.05)

    def _confirm_vendor_actions(self) -> None:
        """Compare the profile's action table with GetActionList (design D7).

        An action the vendor does not report is not advertised: the capability
        parameters shrink to the confirmed subset, so a caller reading
        RuntimeStatus cannot ask for a motion the arm service does not have.
        A query that never answers fails closed the same way.
        """
        table = {str(name): dict(entry) for name, entry in (self._vendor.get("arm_actions") or {}).items()}
        result = self.rpc_call("arm", projection.ARM_ACTION_API_IDS["get_action_list"], {})
        confirmed: set[str] = set()
        if result.ok and isinstance(result.data, dict):
            reported = {str(name) for name in result.data.get("names", [])}
            for name, entry in table.items():
                if str(entry.get("action_name", "")) in reported:
                    confirmed.add(name)
                else:
                    self.get_logger().warning(
                        f"vendor action list does not contain {entry.get('action_name')!r}; "
                        f"motion {name!r} is not advertised"
                    )
        else:
            self.get_logger().warning("GetActionList did not answer; no preset arm actions are advertised")
        with self._lock:
            self._confirmed_actions = confirmed
        self._state.set_capability_params("motion.named", {"names": sorted(confirmed)})

    # --- vendor feedback -------------------------------------------------------

    def _on_lowstate(self, message: vg.LowState) -> None:
        try:
            positions = projection.low_state_joint_positions(vg.lowstate_q(message))
            imu = projection.imu_payload(vg.lowstate_imu(message))
        except ValueError as exc:  # ProjectionError is a ValueError; keep the last good sample
            self.get_logger().warning(f"dropping malformed LowState frame: {exc}", throttle_duration_sec=10.0)
            return
        clock = self.get_clock().now()
        with self._lock:
            self._lowstate_positions = positions
            self._lowstate_stamp = time.monotonic()
            self._lowstate_clock = clock
            self._mode_machine_observed = vg.lowstate_mode_machine(message)
            self._imu_payload = imu

    def _on_hand_state(self, side: str, message: vg.HandState) -> None:
        try:
            positions = projection.dex3_joint_positions(vg.handstate_q(message), side=side)
        except ValueError as exc:  # a wrong slot count or a non-finite position
            self.get_logger().warning(f"dropping malformed {side} hand state: {exc}", throttle_duration_sec=10.0)
            return
        with self._lock:
            self._hand_positions[side] = positions
            self._hand_stamps[side] = time.monotonic()
            self._hand_clocks[side] = self.get_clock().now()

    # --- RPC -------------------------------------------------------------------

    def rpc_call(
        self, service: str, api_id: int, parameter: Any, *, motion: bool = False, timeout_s: float | None = None
    ) -> vg.RpcResult:
        """One request/response round trip, bounded; the entry is always discarded after use.

        The caller blocks in its own callback while the executor's other
        threads deliver the response, so this is only used from service
        handlers, the startup thread and the action server — never from a
        timer (the velocity tick manages its pending call without blocking).
        A timed-out call is a failure; ``motion`` calls are never re-sent here.
        """
        timeout = self._rpc_timeout_s if timeout_s is None else float(timeout_s)
        client = self._rpc[service]
        identity_id = client.send(api_id, parameter, timeout_s=timeout, motion=motion)
        deadline = time.monotonic() + timeout + 0.5
        while True:
            result = client.poll(identity_id)
            if result is not None:
                client.discard(identity_id)
                return result
            if time.monotonic() >= deadline:  # poll resolves its own timeout; this is a hard guard
                client.discard(identity_id)
                return vg.RpcResult(ok=False, code=-1, timed_out=True, motion=motion, api_id=api_id)
            time.sleep(0.01)

    # --- contract services -------------------------------------------------------

    def _on_get_status(self, _request, response):
        response.status = self._state.to_msg(self.get_clock().now().to_msg())
        return response

    def _on_set_mode(self, request, response):
        target = str(request.mode)
        if self._state.stop_latched and target != IDLE_MODE:
            response.success = False
            response.message = f"stop latched ({self._state.stop_policy}); only {IDLE_MODE!r} is accepted"
            response.valid_transitions = [IDLE_MODE]
            return response
        decision = self._modes.can_switch(target)
        if not decision.allowed:
            response.success = False
            response.message = decision.reason
            response.valid_transitions = sorted(self._modes.valid_transitions())
            return response
        if decision.reason == "already active":
            if self._state.stop_latched and target == IDLE_MODE:
                # A stop leaves the runtime in idle with the latch engaged, so
                # requesting idle again is how the latch is cleared.
                if not self._clear_stop_latch():
                    response.success = False
                    response.message = "stop still in progress; request 'idle' again once it has completed"
                    response.valid_transitions = [IDLE_MODE]
                    return response
                response.success = True
                response.message = "stop latch cleared"
                return response
            response.success = True
            response.message = "already active"
            return response
        if self._state.lifecycle != LIFECYCLE_ACTIVE:
            response.success = False
            response.message = (
                f"runtime is {self._state.lifecycle}; mode switches are only admitted while ACTIVE "
                f"({'; '.join(self._state.faults) or 'startup checks still running'})"
            )
            response.valid_transitions = sorted(self._modes.valid_transitions())
            return response
        epoch = self._state.stop_epoch
        ok, detail = self._switch_vendor_mode(target, epoch)
        if not ok:
            response.success = False
            response.message = detail
            response.valid_transitions = (
                [IDLE_MODE] if self._state.stop_latched else sorted(self._modes.valid_transitions())
            )
            return response
        # The commit happens under the stop lock: a stop that engaged while the
        # vendor was confirming must not find the mode half-switched.
        if not self._stop_lock.acquire(blocking=False):
            response.success = False
            response.message = f"stop engaged while switching to {target!r}; request 'idle' to clear the latch"
            response.valid_transitions = [IDLE_MODE]
            return response
        try:
            if self._state.stop_epoch != epoch:
                response.success = False
                response.message = f"stop engaged while switching to {target!r}; request 'idle' to clear the latch"
                response.valid_transitions = [IDLE_MODE]
                return response
            self._modes.commit(target)
            with self._lock:
                self._twist = None
                self._twist_stamp = 0.0
        finally:
            self._stop_lock.release()
        self._publish_status()
        response.success = True
        response.message = detail
        return response

    def _on_stop(self, request, response):
        policy = str(request.policy or STOP_POLICIES[0])
        response.cancel_latency_s = -1.0
        response.idle_latency_s = -1.0
        response.torque_off_latency_s = -1.0
        if policy not in STOP_POLICIES:
            response.success = False
            response.message = f"unknown stop policy {policy!r}; supported: {list(STOP_POLICIES)}"
            return response
        # The posture test needs the vendor's own FSM report, and only for the
        # policy that releases torque: HOLD switches no FSM.
        fsm_id: int | None = None
        if policy == "TORQUE_OFF":
            fsm_id = self._reported_fsm_id()
            if fsm_id is None:
                response.success = False
                response.message = "the reported FSM could not be read; refusing TORQUE_OFF (fail closed)"
                return response
        try:
            plan = projection.resolve_stop_plan(policy, fsm_id=fsm_id, stop_config=self._vendor.get("stop"))
        except CommandRejected as rejected:
            # A balancing biped cannot always honour TORQUE_OFF. Refusing is
            # the honest answer; holding silently and reporting success is not.
            response.success = False
            response.message = f"{rejected.reason}: {rejected.detail}"
            return response

        with self._stop_lock:
            started = time.monotonic()
            # (1) Close admission first, atomically: the latch and the idle
            #     commit stop the input callbacks and the output ticks before
            #     anything else happens, and outlive a vendor failure below.
            self._state.engage_stop(policy)
            self._modes.commit(IDLE_MODE)
            with self._lock:
                self._twist = None
                self._twist_stamp = 0.0
                self._weight_target = plan.weight_target
            response.cancel_latency_s = time.monotonic() - started

            # (2) Stop walking: an explicit zero, confirmed by the vendor.
            zero = self.rpc_call(
                "loco",
                projection.LOCO_API_IDS["set_velocity"],
                projection.velocity_parameter(0.0, 0.0, 0.0, self._velocity_duration_s),
                motion=True,
            )
            if not zero.ok:
                response.idle_latency_s = time.monotonic() - started
                response.success = False
                response.message = (
                    f"SetVelocity(0,0,0) {'timed out' if zero.timed_out else f'failed with code {zero.code}'}; "
                    "stop latched locally, the zero-velocity stop is unconfirmed"
                )
                return response

            # (3) The latched targets are already frozen (admission is closed),
            #     so what remains is handing the frame back: ramp the weight to 0.
            ramped, ramp_detail = self._await_weight_zero()
            response.idle_latency_s = time.monotonic() - started
            if not ramped:
                response.success = False
                response.message = f"{ramp_detail}; stop latched locally, the weight handover is unconfirmed"
                return response

            # (4) TORQUE_OFF releases torque through the damping FSM — never
            #     the zero-torque FSM a caller might imagine.
            if plan.fsm_id is not None:
                release = self.rpc_call(
                    "loco",
                    projection.LOCO_API_IDS["set_fsm_id"],
                    projection.fsm_id_parameter(plan.fsm_id),
                    motion=True,
                )
                response.torque_off_latency_s = time.monotonic() - started
                if not release.ok:
                    response.success = False
                    response.message = (
                        f"SetFsmId({plan.fsm_id}) "
                        f"{'timed out' if release.timed_out else f'failed with code {release.code}'}; "
                        "stop latched locally, the torque release is unconfirmed"
                    )
                    return response

        self._publish_status()
        response.success = True
        response.message = (
            "stopped: admission closed, velocity zeroed, streaming targets frozen, shared weight ramped to 0"
        )
        if plan.fsm_id is not None:
            response.message += (
                f"; torque released through the damping FSM (requested {plan.downgraded_from}; "
                "the platform-safe equivalent on a balancing biped)"
            )
        return response

    def _reported_fsm_id(self) -> int | None:
        """The vendor's own FSM report, or None when it does not answer."""
        result = self.rpc_call(
            "loco", projection.LOCO_API_IDS["get_fsm_id"], {}, timeout_s=min(self._rpc_timeout_s, 1.0)
        )
        if result.ok and isinstance(result.data, dict):
            try:
                return int(result.data["data"])
            except (KeyError, TypeError, ValueError):
                return None
        return None

    def _clear_stop_latch(self) -> bool:
        """Clear the latch, unless a stop is still holding it."""
        if not self._stop_lock.acquire(blocking=False):
            return False
        try:
            self._state.clear_stop()
            if self._startup_fault is not None:
                # A stop never healed a generation mismatch.
                self._state.set_status("FAULTED", [self._startup_fault])
        finally:
            self._stop_lock.release()
        return True

    # --- interaction services ---------------------------------------------------

    def _on_speak(self, request, response):
        """SpeakText -> the vendor TTS call, refusing what it cannot map.

        The vendor parameter carries only index/speaker_id/text: language,
        priority and interrupt have no counterpart, so a request that sets
        them is refused rather than silently dropped on the floor. trace_id is
        "propagated where supported" per the contract and is not, here.
        """
        response.utterance_id = ""
        if self._state.stop_latched:
            response.success, response.error_code = False, "STOP_LATCHED"
            response.message = "stop latched"
            return response
        text = str(request.text)
        if not text.strip():
            response.success, response.error_code = False, "INVALID_REQUEST"
            response.message = "text is empty"
            return response
        if str(request.language or ""):
            response.success, response.error_code = False, "UNSUPPORTED_LANGUAGE"
            response.message = (
                f"the vendor TTS call carries no language parameter; {request.language!r} cannot be honoured"
            )
            return response
        if int(request.priority) != 0:
            response.success, response.error_code = False, "INVALID_REQUEST"
            response.message = "the vendor TTS call carries no priority; only the neutral default 0 maps"
            return response
        if bool(request.interrupt):
            response.success, response.error_code = False, "INVALID_REQUEST"
            response.message = "the vendor TTS call has no interrupt semantics; requests queue at the vendor"
            return response
        tts = self._vendor.get("tts") or {}
        result = self.rpc_call(
            "audio",
            projection.AUDIO_API_IDS["tts"],
            projection.tts_parameter(text, index=int(tts.get("index", 0)), speaker_id=int(tts.get("speaker_id", 0))),
        )
        response.success = result.ok
        response.error_code = "" if result.ok else "REJECTED"
        response.message = (
            "spoken through the vendor TTS call"
            if result.ok
            else (
                "the vendor TTS call timed out"
                if result.timed_out
                else f"the vendor TTS call failed with vendor code {result.code}"
            )
        )
        return response

    def _on_led(self, request, response):
        """SetLedPattern -> the vendor RGB call, refusing what it cannot map.

        SetRgbLed sets one static colour: only PATTERN_SOLID maps, and the
        call carries no priority or preemption, so those must stay neutral.
        active_priority stays 0 even on success — the vendor reports no
        effective priority, and echoing the request would claim one it won.
        """
        response.active_priority = 0
        if self._state.stop_latched:
            response.success, response.error_code = False, "STOP_LATCHED"
            response.message = "stop latched"
            return response
        if int(request.pattern) != SetLedPattern.Request.PATTERN_SOLID:
            response.success, response.error_code = False, "UNSUPPORTED_PATTERN"
            response.message = "the vendor LED call sets one static colour; only PATTERN_SOLID maps"
            return response
        if int(request.priority) != 0 or bool(request.preempt):
            response.success, response.error_code = False, "INVALID_REQUEST"
            response.message = "the vendor LED call carries no priority or preemption; only the neutral defaults map"
            return response
        result = self.rpc_call(
            "audio",
            projection.AUDIO_API_IDS["set_rgb_led"],
            projection.led_parameter(int(request.r), int(request.g), int(request.b)),
        )
        response.success = result.ok
        response.error_code = "" if result.ok else "REJECTED"
        response.message = (
            "set through the vendor RGB call"
            if result.ok
            else (
                "the vendor LED call timed out"
                if result.timed_out
                else f"the vendor LED call failed with vendor code {result.code}"
            )
        )
        return response

    # --- mode switches (design D5) ------------------------------------------------

    def _switch_vendor_mode(self, mode: str, epoch: int) -> tuple[bool, str]:
        """Perform the vendor side of a mode switch. Returns (ok, detail).

        Every wait inside is interruptible by a stop (the ``epoch`` moves), so
        a stop never queues behind a mode switch that is waiting on the vendor.
        The detail names each vendor action performed, in the order performed.
        """
        current = self._modes.mode
        steps: list[str] = []
        if current == "stream" and mode != "stream":
            ok, detail = self._exit_stream(epoch)
            if not ok:
                return False, detail
            steps.append(detail)
        if current == "locomotion" and mode != "locomotion":
            ok, detail = self._leave_locomotion(epoch)
            if not ok:
                return False, detail
            steps.append(detail)
        if mode in ("stream", "locomotion"):
            entered = self._enter_stream(epoch) if mode == "stream" else self._enter_locomotion(epoch)
            if not entered[0]:
                return False, entered[1]
            steps.append(entered[1])
        elif mode != IDLE_MODE:
            return False, f"mode {mode!r} has no vendor-side action"
        if not steps:
            steps.append("the vendor's internal control keeps the frame (arm_sdk weight 0)")
        return True, "; ".join(steps)

    def _leave_locomotion(self, epoch: int) -> tuple[bool, str]:
        """Zero the base and have the vendor confirm it before leaving locomotion.

        The profile's re-issue semantics mean the last SetVelocity alone keeps
        its effect for its duration: nothing re-issues it after the switch, so
        without an explicit zero the robot would keep walking for up to
        command_duration_s in a mode that no longer admits base commands. A
        zero the vendor does not confirm fails the switch — the mode must not
        change while the base is still commanded.
        """
        with self._lock:
            self._twist = None
            self._twist_stamp = 0.0
        result = self.rpc_call(
            "loco",
            projection.LOCO_API_IDS["set_velocity"],
            projection.velocity_parameter(0.0, 0.0, 0.0, self._velocity_duration_s),
            motion=True,
        )
        if self._state.stop_epoch != epoch:
            return False, "stop engaged while the mode switch was waiting on the vendor"
        if not result.ok:
            detail = "timed out; not re-sent" if result.timed_out else f"failed with vendor code {result.code}"
            return False, f"SetVelocity(0,0,0) {detail}; refusing to leave locomotion with the base still commanded"
        with self._lock:
            self._velocity_sent = (0.0, 0.0, 0.0)
            self._last_velocity_sent_s = time.monotonic()
        return True, "locomotion left: the base velocity was zeroed and the vendor confirmed it"

    def _enter_stream(self, epoch: int) -> tuple[bool, str]:
        """Fresh feedback -> latch the 17-joint hold -> SwitchToUserCtrl -> ramp up (D4)."""
        with self._lock:
            stamp = self._lowstate_stamp
            positions = dict(self._lowstate_positions)
        if not projection.is_fresh(stamp, time.monotonic(), self._feedback_window_s):
            return False, (
                f"FEEDBACK_STALE: the LowState feed is older than the profile's "
                f"feedback_staleness_s ({self._feedback_window_s}s); refusing to latch hold targets"
            )
        hold = projection.capture_hold_targets(positions, order=projection.ARM_SDK_FRAME)
        if hold is None:
            return False, (
                "HOLD_TARGETS_INCOMPLETE: the measured positions do not cover the 17-joint arm_sdk "
                "frame; a partial measurement is no basis for a hold target"
            )
        result = self.rpc_call("loco", projection.LOCO_API_IDS["switch_to_user_ctrl"], {}, motion=True)
        if self._state.stop_epoch != epoch:
            return False, "stop engaged while the mode switch was waiting on the vendor"
        if not result.ok:
            detail = "timed out; not re-sent" if result.timed_out else f"failed with vendor code {result.code}"
            return False, f"SwitchToUserCtrl {detail}"
        with self._lock:
            self._latch = dict(zip(projection.ARM_SDK_FRAME, hold, strict=True))
            # A fresh stream session starts with no hand targets: a target from
            # a previous session would be a motion nobody commanded.
            self._hand_latch = {"left": {}, "right": {}}
            self._weight_target = projection.MODE_WEIGHT_TARGETS["stream"]
        return True, "stream entered: hold targets latched, SwitchToUserCtrl confirmed, weight ramping to 1"

    def _exit_stream(self, epoch: int) -> tuple[bool, str]:
        """Ramp the shared weight back to 0 before the mode is reported inactive (D4)."""
        with self._lock:
            self._weight_target = projection.MODE_WEIGHT_TARGETS[IDLE_MODE]
        ramped, detail = self._await_weight_zero()
        if ramped:
            return True, "stream exited: the shared weight reached 0 and the vendor's controller took over"
        if self._state.stop_epoch != epoch:
            return False, "stop engaged while stream was exiting"
        return False, detail

    def _await_weight_zero(self) -> tuple[bool, str]:
        """Wait (bounded) for the 50 Hz tick to ramp the shared weight to 0."""
        with self._lock:
            weight = self._weight
        if weight <= 0.0:
            return True, ""
        if self._weight_ramp_down_rate <= 0.0:
            return False, "the profile's weight ramp-down rate is not positive"
        # At the stop section's rate the ramp cannot take longer than
        # weight/rate; the margin only covers timer jitter.
        deadline = time.monotonic() + weight / self._weight_ramp_down_rate + 1.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._weight <= 0.0:
                    return True, ""
            time.sleep(0.02)
        return False, "the shared weight did not ramp back to 0 within the expected time"

    def _enter_locomotion(self, epoch: int) -> tuple[bool, str]:
        """SetFsmId(start), then commit only once GetFsmId reports it (D5)."""
        fsm_id = projection.FSM_IDS["start"]
        result = self.rpc_call(
            "loco",
            projection.LOCO_API_IDS["set_fsm_id"],
            projection.fsm_id_parameter(fsm_id),
            motion=True,
        )
        if self._state.stop_epoch != epoch:
            return False, "stop engaged while the mode switch was waiting on the vendor"
        if not result.ok:
            detail = "timed out; not re-sent" if result.timed_out else f"failed with vendor code {result.code}"
            return False, f"SetFsmId({fsm_id}) {detail}"
        deadline = time.monotonic() + self._rpc_timeout_s
        while time.monotonic() < deadline:
            if self._state.stop_epoch != epoch:
                return False, "stop engaged while waiting for the vendor to confirm the FSM"
            answer = self.rpc_call("loco", projection.LOCO_API_IDS["get_fsm_id"], {}, timeout_s=0.5)
            if answer.ok and isinstance(answer.data, dict) and answer.data.get("data") == fsm_id:
                return True, f"locomotion confirmed: the vendor reports FSM {fsm_id}"
            time.sleep(0.05)
        return False, f"the vendor did not report FSM {fsm_id} within {self._rpc_timeout_s:.0f}s"

    # --- command channels ------------------------------------------------------

    def _on_channel(self, channel: dict[str, Any], message: Float64MultiArray) -> None:
        """Merge one streaming channel's payload into its latch (design D4).

        A channel that goes quiet simply stops updating the latch, so its slots
        keep the last commanded value — the profile's stream_staleness_s
        policy holds by construction, with no zero fill and no measured
        refill.
        """
        name = str(channel["channel"])
        if not self._channel_admitted(channel):
            self._modes.note_rejected(name)
            self._publish_status()
            return
        joints = [str(joint) for joint in channel.get("joints", [])]
        try:
            targets = projection.channel_targets(
                message.data, channel_joints=joints, joint_limits=self._joint_limits, label=name
            )
        except CommandRejected as rejected:
            self._reject_channel_command(name, rejected.reason, str(rejected))
            return
        except ProjectionError as error:
            self._reject_channel_command(name, "PROFILE_ERROR", str(error))
            return
        side = self._hand_side(joints)
        if side is None and not set(joints) <= set(projection.ARM_SDK_FRAME):
            self._reject_channel_command(
                name, "UNKNOWN_CHANNEL_JOINTS", "the channel names joints outside the arm_sdk frame and the Dex3 hands"
            )
            return
        with self._lock:
            if side is not None:
                self._hand_latch[side] = projection.merge_latch(self._hand_latch[side], targets)
            else:
                self._latch = projection.merge_latch(self._latch, targets)

    def _on_twist(self, channel: dict[str, Any], message: Twist) -> None:
        name = str(channel["channel"])
        if not self._channel_admitted(channel):
            self._modes.note_rejected(name)
            self._publish_status()
            return
        try:
            triple = projection.twist_to_velocity(
                message.linear.x, message.linear.y, message.angular.z, self._locomotion_limits
            )
        except CommandRejected as rejected:
            self._reject_channel_command(name, rejected.reason, str(rejected))
            return
        with self._lock:
            self._twist = triple
            self._twist_stamp = time.monotonic()

    def _reject_channel_command(self, channel: str, reason: str, detail: str) -> None:
        self._modes.note_rejected(f"{channel}:{reason}")
        self.get_logger().warning(f"rejected {channel} command: {detail}")
        self._publish_status()

    def _channel_admitted(self, channel: dict[str, Any]) -> bool:
        """Input-side admission; the output path re-checks on every tick."""
        modes = [str(mode) for mode in channel.get("modes", [])]
        if not modes or self._modes.mode not in modes:
            return False
        return self._runtime_admitted()

    def _runtime_admitted(self) -> bool:
        """Whether command input and output are admitted at all right now."""
        return self._state.lifecycle == LIFECYCLE_ACTIVE and not self._state.stop_latched

    @staticmethod
    def _hand_side(channel_joints: list[str]) -> str | None:
        """Which Dex3 hand a channel commands, or None when it is not a hand channel."""
        if not channel_joints:
            return None
        for side in ("left", "right"):
            if set(channel_joints) <= set(projection.dex3_joint_names(side)):
                return side
        return None

    # --- output ticks -------------------------------------------------------------

    def _arm_sdk_tick(self) -> None:
        """Publish the 17-joint frame at the profile's arm_sdk rate (design D4).

        Admission is re-checked here, on the output path, not only when a
        command arrived: a stop, a fault or a mode change takes effect on the
        very next tick. The one command admitted outside stream mode is the
        weight ramp-down itself — that is the stop/exit handover, and its
        latch is frozen because admission is closed.
        """
        with self._lock:
            latch = dict(self._latch)
            weight = self._weight
            target = self._weight_target
        if len(latch) != projection.ARM_SDK_FRAME_SIZE:
            return  # never entered stream: there is no frame to hand over
        if not (self._modes.spec().allows_stream and self._runtime_admitted()):
            target = min(target, projection.MODE_WEIGHT_TARGETS[IDLE_MODE])
        if weight <= 0.0 and target <= 0.0:
            return  # idle: the vendor's internal control owns the frame
        # Ramp-up hands the frame over at the arm_sdk rate; every handover
        # back (stop, exit, closed admission) uses the stop section's rate.
        rate = self._weight_ramp_up_rate if target > weight else self._weight_ramp_down_rate
        new_weight = projection.ramp_weight(weight, target, rate_per_s=rate, dt_s=self._arm_sdk_period_s)
        message = vg.lowcmd_from_dict(
            projection.arm_sdk_lowcmd(
                latch,
                weight=new_weight,
                mode_pr=self._mode_pr,
                mode_machine=self._mode_machine_expected,
                kp=self._arm_sdk_kp,
                kd=self._arm_sdk_kd,
            )
        )
        with self._lock:
            self._weight = new_weight
        self._arm_sdk_pub.publish(message)

    def _dex3_tick(self) -> None:
        """Publish latched hand targets at the profile's Dex3 rate, in stream mode."""
        if not (self._modes.spec().allows_stream and self._runtime_admitted()):
            return
        with self._lock:
            targets = {side: dict(latch) for side, latch in self._hand_latch.items()}
        for side, latched in targets.items():
            if not latched:
                continue  # a hand nobody commanded is not commanded at all
            self._hand_pubs[side].publish(
                vg.handcmd_from_dict(projection.dex3_handcmd(latched, side=side, kp=self._dex3_kp, kd=self._dex3_kd))
            )

    def _velocity_tick(self) -> None:
        """Issue SetVelocity for the current command at the profile's duration.

        A command that differs from the last one sent goes out at once —
        waiting out the previous command's duration would delay a stop or a
        change of direction by up to that duration. The duration throttle
        applies only to repeating the SAME command, which is the renewal the
        profile describes. One call is in flight at a time; while a call is
        pending the latest command waits in the cache and is dispatched on a
        later tick the moment the pending one resolves and the comparison
        still shows a difference.
        """
        with self._lock:
            pending = self._velocity_pending
            twist = self._twist
            twist_stamp = self._twist_stamp
            last_sent = self._last_velocity_sent_s
            sent = self._velocity_sent
        now = time.monotonic()
        if pending is not None:
            result = self._rpc["loco"].poll(pending)
            if result is None:
                return  # still in flight; the latest command is already cached
            self._rpc["loco"].discard(pending)
            with self._lock:
                self._velocity_pending = None
            if not result.ok:
                # A motion call whose fate is unknown is not re-sent: the
                # cached command is abandoned and only a fresh upstream Twist
                # re-arms the cadence.
                with self._lock:
                    self._twist = None
                    self._twist_stamp = 0.0
                    self._velocity_sent = None
                self.get_logger().warning(
                    f"SetVelocity {'timed out' if result.timed_out else f'failed with code {result.code}'}; "
                    "the cached velocity command was dropped, a fresh Twist re-arms it"
                )
                return
        if not (self._modes.spec().allows_base and self._runtime_admitted()):
            return
        if twist is None or now - twist_stamp > self._twist_staleness_s:
            return
        if twist == sent and now - last_sent < self._velocity_duration_s:
            return  # the same command is still within its issued duration
        identity_id = self._rpc["loco"].send(
            projection.LOCO_API_IDS["set_velocity"],
            projection.velocity_parameter(twist[0], twist[1], twist[2], self._velocity_duration_s),
            timeout_s=self._rpc_timeout_s,
            motion=True,
        )
        with self._lock:
            self._velocity_pending = identity_id
            self._velocity_sent = twist
            self._last_velocity_sent_s = now

    # --- state publication --------------------------------------------------------

    def _publish_joint_states(self) -> None:
        """Publish /joint_states and /imu/data at the profile's declared rate.

        A source that went quiet drops out of the aggregate instead of being
        re-stamped, and joints the vendor feeds do not carry are omitted,
        never zero-filled (a fabricated zero reads as a real measurement
        downstream). The stamp is the oldest receipt in the aggregate.
        """
        now = time.monotonic()
        with self._lock:
            measurements: dict[str, float] = {}
            clocks: list[Any] = []
            lowstate_clock = None
            imu = None
            if projection.is_fresh(self._lowstate_stamp, now, self._feedback_window_s):
                measurements.update(self._lowstate_positions)
                lowstate_clock = self._lowstate_clock
                clocks.append(lowstate_clock)
                imu = self._imu_payload
            for side in ("left", "right"):
                if projection.is_fresh(self._hand_stamps[side], now, self._feedback_window_s):
                    measurements.update(self._hand_positions[side])
                    clocks.append(self._hand_clocks[side])
        names, positions = projection.contract_joint_state(measurements, order=self._joint_order)
        if names and clocks:
            message = JointState()
            message.header.stamp = min(clocks).to_msg()
            message.name = names
            message.position = positions
            self._joint_pub.publish(message)
        if imu is not None and lowstate_clock is not None:
            republished = Imu()
            republished.header.stamp = lowstate_clock.to_msg()
            republished.orientation.x = float(imu["orientation"]["x"])
            republished.orientation.y = float(imu["orientation"]["y"])
            republished.orientation.z = float(imu["orientation"]["z"])
            republished.orientation.w = float(imu["orientation"]["w"])
            republished.angular_velocity.x = float(imu["angular_velocity"]["x"])
            republished.angular_velocity.y = float(imu["angular_velocity"]["y"])
            republished.angular_velocity.z = float(imu["angular_velocity"]["z"])
            republished.linear_acceleration.x = float(imu["linear_acceleration"]["x"])
            republished.linear_acceleration.y = float(imu["linear_acceleration"]["y"])
            republished.linear_acceleration.z = float(imu["linear_acceleration"]["z"])
            self._imu_pub.publish(republished)

    def _publish_status(self) -> None:
        self._status_pub.publish(self._state.to_msg(self.get_clock().now().to_msg()))

    # --- accessors used by helper modules -------------------------------------------

    @property
    def profile(self) -> dict[str, Any]:
        return self._profile

    @property
    def runtime_state(self) -> RuntimeState:
        return self._state

    @property
    def vendor_config(self) -> dict[str, Any]:
        return self._vendor

    @property
    def action_callback_group(self) -> Any:
        """Long-running action execution must not block the mode/stop services."""
        return self._action_cb

    def current_mode(self) -> str:
        return self._modes.mode

    def served_motion_names(self) -> set[str]:
        """Postures plus the arm actions the vendor's own list confirmed."""
        with self._lock:
            confirmed = set(self._confirmed_actions)
        return confirmed | set(self._vendor.get("postures") or {})


def main(argv=None) -> int:
    # SIGTERM included so `timeout`/systemd stops exit spin via the ordinary
    # ExternalShutdownException path instead of dying inside an rcl wait set.
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.ALL)
    node = UnitreeG1RuntimeNode()
    # Mode switches and stops block their own callbacks for the weight ramp
    # and vendor confirmations; the output ticks must keep running meanwhile,
    # so the executor needs more threads than the default one-per-callback-group.
    executor = MultiThreadedExecutor(num_threads=8)
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
