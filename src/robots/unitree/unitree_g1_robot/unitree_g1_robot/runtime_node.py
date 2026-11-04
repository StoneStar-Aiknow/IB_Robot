"""Unitree G1 runtime: the public runtime contract served over the official SDK.

This is the wrapper layer. Generic IB-Robot packages talk only to contract
names (``/runtime_status``, ``/runtime/set_mode``, ``/runtime/get_status``,
``/runtime/stop``, ``/joint_states``, ``/imu/data`` and the profile's declared
command channels) and never import a vendor symbol; the vendor's topics, FSM
ids and joint order stay behind this node.

Vendor integration (Issue #171 PR 2): the official ``unitree_sdk2py`` is the
transport, reached only through the ROS-free adapter ``unitree_g1_adapter``.
The vendor channel factory is initialized once per process through the
adapter's ``UnitreeSdk`` facade (a conflicting second initialization fails
loudly there); state feedback rides ``StateFeed`` subscribers (last-sample
cached on the SDK's reader threads); the 17-joint arm_sdk frame and the Dex3
hand frames go out through ``CommandWriter`` publishers; and the vendor RPCs
run on the SDK's own typed clients (``LocoClient``/``G1ArmActionClient``/
``AudioClient``), whose mixed return shapes the adapter normalizes onto
``SdkCallResult``.

What the node does NOT do (design D1/D11): instantiate a controller manager,
load a ros2_control hardware component, or publish on the unprotected
``rt/lowcmd`` tier — ``expert_tier_gate`` refuses a profile that enables it.

Key behaviours, per openspec design:

* D4 shared-weight frame: entering stream latches fresh measured positions as
  hold targets for all 17 arm_sdk joints; read-only admission protects the frame
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
* Stop/blocking-RPC contract (Issue #171 Q7): stop latches admission closed
  immediately, discards every unsent command, and NEVER re-sends a motion
  request whose outcome is unknown — the SDK client does not retry and the
  call sites honour that. Voice/LED calls run in their own callback group, so
  a slow or timed-out TTS call cannot delay a motion stop. The SDK RPC client
  objects are wrapped by one serialized dispatcher per client (the SDK's
  clients are plain request/future objects without a documented thread-safety
  guarantee), so concurrency stays inside this class.
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

import contextlib
import json
import threading
import time
from collections.abc import Callable
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
from unitree_g1_adapter.runtime import UnitreeSdk

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

#: SetVelocity re-issue cadence check. The command itself carries the
#: profile's duration; this tick only decides when a re-issue is due, and a
#: tenth of that duration keeps the re-issue prompt without spamming the tier.
_VELOCITY_TICK_S = 0.1

#: Visible warning for an arm handover the platform cannot confirm. The local
#: side — admission closed, latched targets frozen, published weight at 0 — is
#: observable and proven; whether the vendor then hands the frame back to its
#: own control is not, and the runtime keeps saying so in ``RuntimeStatus``
#: instead of claiming a stop it cannot see. ``SetRuntimeMode("idle")`` clears
#: it once the operator has taken the frame back (see ``_clear_stop_latch``).
ARM_HANDOVER_WARNING = (
    "ARM_HANDOVER_UNCONFIRMED: the zero-weight frame was published locally; the vendor's own arm "
    "handover is not acknowledged by this interface"
)

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


def vendor_dds_binding(domain_id: int, network_interface: str) -> tuple[int, str]:
    """The vendor DDS (domain, NIC) pair, or a refusal to open the vendor tier.

    The two halves of the binding only make sense together, so the pair — not
    each half — is what is enforced. A nonzero domain is the isolated one
    (simulation, tests) and may only bind loopback, so its traffic cannot reach
    the robot's production graph. Domain 0 *is* the robot's graph and therefore
    needs the explicit robot-facing NIC: binding it to loopback, or letting the
    SDK autodetermine, would silently open the vendor tier on whatever interface
    the host happens to have. There is no safe default, so an unset pair fails
    closed instead of guessing — the launch entry always passes both, and the
    mock enforces the same rule because a fake vendor on the robot's production
    domain is exactly what a test double must never become.
    """
    domain_id = int(domain_id)
    network_interface = str(network_interface or "").strip()
    if domain_id != 0:
        if network_interface != "lo":
            raise RuntimeError(
                f"VENDOR_DDS_ISOLATION_REQUIRED: vendor domain {domain_id} is an isolated domain and "
                f"must bind loopback ('lo'), got {network_interface!r}"
            )
        return domain_id, network_interface
    if not network_interface:
        raise RuntimeError(
            "VENDOR_DDS_INTERFACE_REQUIRED: vendor domain 0 is the robot's production domain and needs an "
            "explicit vendor_network_interface naming the robot-facing NIC"
        )
    if network_interface == "lo":
        raise RuntimeError(
            "VENDOR_DDS_ISOLATION_REQUIRED: vendor domain 0 must not bind loopback; name the robot-facing "
            "NIC, or use a nonzero isolated domain"
        )
    return domain_id, network_interface


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
        # through checked admission (stream) or SetFsmId (locomotion), so a
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
        self._preset_only = self._vendor.get("preset_only") is True
        self.declare_parameter("preset_test_transport", False)
        self.declare_parameter("preset_operator_confirmed", False)
        self._preset_physical = self._preset_only and self.get_parameter("preset_test_transport").value is not True
        # Explicit, operator-supervised commissioning. The shipped
        # profile stays disabled; simulation still requires loopback isolation.
        if self._preset_only:
            if self._preset_physical and self.get_parameter("preset_operator_confirmed").value is not True:
                raise RuntimeError("PRESET_HARDWARE_DISABLED: on-site operator confirmation required")
            if self._profile.get("command_channels") or self._vendor.get("postures"):
                raise RuntimeError("preset-only profile cannot expose streams or postures")
            allowed = (self._vendor.get("preset_commissioning") or {}).get("allowed_arm_actions")
            if not allowed or self._vendor.get("arm_actions") != allowed:
                raise RuntimeError(
                    "preset-only commissioning permits only the profile's preset_commissioning.allowed_arm_actions"
                )
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
        # The vendor channel factory is a process-wide singleton; its binding
        # (domain, NIC) is deployment configuration, so the launch entry passes
        # it as parameters. The pair is enforced for EVERY mode, not only the
        # preset surface: without it the SDK falls back to DDS autodetermination
        # on the robot's production domain (domain 0, any NIC), and
        # ``CYCLONEDDS_URI`` cannot constrain the result because the SDK hands
        # its own configuration to ``Domain(id, config)``, which overrides it.
        self.declare_parameter("vendor_domain_id", 0)
        self.declare_parameter("vendor_network_interface", "")
        self._vendor_domain_id, self._vendor_network_interface = self._vendor_dds_binding()

        # --- shared mutable state (guarded by _lock) -------------------------
        self._lock = threading.RLock()
        # The 17-joint arm_sdk latch: seeded from fresh measurements at stream
        # entry, updated only by admitted channel messages after that.
        self._latch: dict[str, float] = {}
        self._hand_latch: dict[str, dict[str, float]] = {"left": {}, "right": {}}
        self._weight = 0.0
        self._arm_write_fault: str | None = None
        self._arm_write_pending = False
        self._arm_handover_unconfirmed = False
        self._weight_target = projection.MODE_WEIGHT_TARGETS[IDLE_MODE]
        self._twist: tuple[float, float, float] | None = None
        self._twist_stamp = 0.0
        self._velocity_inflight = False
        self._locomotion_requested = False
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
        self._tick_cb = MutuallyExclusiveCallbackGroup()
        self._state_cb = MutuallyExclusiveCallbackGroup()
        self._service_cb = MutuallyExclusiveCallbackGroup()
        self._stop_cb = ReentrantCallbackGroup()
        self._interaction_cb = MutuallyExclusiveCallbackGroup()
        self._action_cb = ReentrantCallbackGroup()

        self._status_pub = self.create_publisher(RuntimeStatus, STATUS_TOPIC, 10)
        self._joint_pub = self.create_publisher(JointState, str(self._profile["joint_state_topic"]), _RELIABLE_QOS)
        self._imu_pub = self.create_publisher(Imu, IMU_TOPIC, _SENSOR_QOS)

        self._setup_vendor_io()
        self._setup_contract_surface()

        if not self._preset_only:
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

    def _vendor_dds_binding(self) -> tuple[int, str]:
        """The vendor DDS (domain, NIC) pair from this node's parameters."""
        return vendor_dds_binding(
            int(self.get_parameter("vendor_domain_id").value),
            str(self.get_parameter("vendor_network_interface").value or ""),
        )

    def _setup_vendor_io(self) -> None:
        """Open the official SDK's channels/clients through the adapter facade.

        The facade owns the process-wide vendor channel factory singleton; the
        state feeds, command writers and typed RPC clients it hands out are
        all tracked and released by ``UnitreeSdk.close`` at shutdown. When the
        runtime is launched against the in-repo vendor mock (simulated
        transport), the same facade is wired to the mock's injectable surface
        by the entry point, so this method's code is identical either way.
        """
        vg.require_vendor()
        topics = self._vendor["topics"]
        self._sdk = UnitreeSdk(now=time.monotonic)
        self._sdk.initialize(self._vendor_domain_id, self._vendor_network_interface)
        self._feeds: dict[str, Any] = {}
        # queue_len=1: the SDK delivers through its queue thread, so the
        # writer's path never inlines our handler (an in-process writer whose
        # ddspy_write holds the GIL while a python handler re-enters on the
        # same delivery deadlocks both — observed against the in-repo mock).
        self._feeds["low_state"] = self._sdk.state_feed(
            topics["low_state"], "low_state", handler=self._on_lowstate_sample, queue_len=1
        )
        if self._preset_only:
            self._hand_writers = {}
            self._rpc = {"arm": self._sdk.arm_action(timeout_s=self._rpc_timeout_s)}
            if self._preset_physical:
                self._rpc["loco"] = self._sdk.loco(timeout_s=self._rpc_timeout_s)
                self._rpc["robot_state"] = self._sdk.robot_state(timeout_s=self._rpc_timeout_s)
            self._rpc_dispatchers = {name: threading.Lock() for name in self._rpc}
            return
        for side in ("left", "right"):
            self._feeds[f"dex3_{side}_state"] = self._sdk.state_feed(
                topics[f"dex3_{side}_state"],
                "hand_state",
                handler=(lambda sample, side=side: self._on_hand_state_sample(side, sample)),
                queue_len=1,
            )
        self._arm_sdk_writer = self._sdk.command_writer(topics["arm_sdk"], "low_cmd")
        self._hand_writers = {
            side: self._sdk.command_writer(topics[f"dex3_{side}_cmd"], "hand_cmd") for side in ("left", "right")
        }
        # The SDK's typed RPC clients block up to their timeout on the calling
        # thread; they are created eagerly so startup's discovery wait happens
        # once, and every call goes through the serialized dispatcher below.
        self._rpc: dict[str, Any] = {
            "loco": self._sdk.loco(timeout_s=self._rpc_timeout_s),
            "arm": self._sdk.arm_action(timeout_s=self._rpc_timeout_s),
            "audio": self._sdk.audio(timeout_s=self._rpc_timeout_s),
            "robot_state": self._sdk.robot_state(timeout_s=self._rpc_timeout_s),
        }
        self._motion_switcher = self._sdk.motion_switcher(timeout_s=self._rpc_timeout_s)
        # One dispatcher lock per service: the SDK clients have no documented
        # thread-safety guarantee, so calls to one service serialize; calls to
        # different services (e.g. voice while locomotion confirms) stay
        # independent.
        self._rpc_dispatchers: dict[str, threading.Lock] = {name: threading.Lock() for name in self._rpc}

    # --- vendor feedback -------------------------------------------------------

    def _on_lowstate_sample(self, sample: Any) -> None:
        """SDK reader-thread callback: cache the newest LowState (never blocks)."""
        try:
            positions = projection.low_state_joint_positions(vg.lowstate_q(sample))
            imu = projection.imu_payload(vg.lowstate_imu(sample))
        except ValueError as exc:  # ProjectionError is a ValueError; keep the last good sample
            self.get_logger().warning(f"dropping malformed LowState frame: {exc}", throttle_duration_sec=10.0)
            return
        clock = self.get_clock().now()
        with self._lock:
            self._lowstate_positions = positions
            self._lowstate_stamp = time.monotonic()
            self._lowstate_clock = clock
            self._mode_machine_observed = vg.lowstate_mode_machine(sample)
            self._imu_payload = imu

    def _on_hand_state_sample(self, side: str, sample: Any) -> None:
        """SDK reader-thread callback for one hand's state stream (never blocks)."""
        try:
            positions = projection.dex3_joint_positions(vg.handstate_q(sample), side=side)
        except ValueError as exc:  # a wrong slot count or a non-finite position
            self.get_logger().warning(f"dropping malformed {side} hand state: {exc}", throttle_duration_sec=10.0)
            return
        with self._lock:
            self._hand_positions[side] = positions
            self._hand_stamps[side] = time.monotonic()
            self._hand_clocks[side] = self.get_clock().now()

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
                have_hands = self._preset_only or all(self._hand_stamps[side] > 0.0 for side in ("left", "right"))
                observed = self._mode_machine_observed
            if have_lowstate and have_hands:
                break
            if time.monotonic() >= deadline:
                missing: list[str] = []
                if not have_lowstate:
                    missing.append("the first LowState frame")
                with self._lock:
                    quiet = [side for side in ("left", "right") if self._hand_stamps[side] <= 0.0]
                if not self._preset_only:
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

        self._wait_for_rpc_peers()
        self._confirm_vendor_actions()
        self._state.set_lifecycle(LIFECYCLE_ACTIVE)

    def _fail_startup(self, detail: str) -> None:
        """Hold the runtime out of ACTIVE for good: a D10 fault is a generation mismatch."""
        self._startup_fault = detail
        self._state.set_status("FAULTED", [detail])

    def _wait_for_rpc_peers(self) -> None:
        """Wait until the SDK clients can reach their services.

        A request published before DDS matching completes is silently dropped,
        so the startup queries would otherwise observe timeouts that are
        really a discovery race. The SDK client's request channel is created
        inside ``ClientStub.Init`` (which already sleeps 0.5 s), and the
        first bounded query doubles as the match probe: each service gets one
        cheap read-only call whose timeout the wait absorbs.
        """
        probes = {
            "loco": ("get_fsm_id", None),
            "arm": ("get_action_list", None),
            "audio": ("get_volume", None),
        }
        deadline = time.monotonic() + _RPC_MATCH_TIMEOUT_S
        remaining = {key: value for key, value in probes.items() if key in self._rpc}
        while remaining and time.monotonic() < deadline:
            for service, (api_name, parameter) in list(remaining.items()):
                result = self.rpc_call(service, api_name, parameter, timeout_s=0.5)
                if not result.timed_out:
                    remaining.pop(service)
            if remaining:
                time.sleep(0.05)
        for service in remaining:
            self.get_logger().warning(
                f"the vendor {service!r} service did not answer its startup probe within "
                f"{_RPC_MATCH_TIMEOUT_S:.0f}s; proceeding (calls will surface their own timeouts)"
            )

    def _confirm_vendor_actions(self) -> None:
        """Compare the profile's action table with GetActionList (design D7).

        An action the vendor does not report is not advertised: the capability
        parameters shrink to the confirmed subset, so a caller reading
        RuntimeStatus cannot ask for a motion the arm service does not have.
        A query that never answers fails closed the same way.
        """
        table = {str(name): dict(entry) for name, entry in (self._vendor.get("arm_actions") or {}).items()}
        result = self.rpc_call("arm", "get_action_list", None)
        confirmed: set[str] = set()
        if result.ok and isinstance(result.data, list):
            # Actual firmware returns [preset objects, custom-action objects].
            presets = result.data[0] if result.data and isinstance(result.data[0], list) else result.data
            for name, entry in table.items():
                if any(
                    isinstance(item, dict)
                    and type(item.get("id")) is int
                    and item["id"] == entry.get("action_id")
                    and item.get("name") == entry.get("action_name")
                    for item in presets
                ):
                    confirmed.add(name)
        elif result.ok and isinstance(result.data, dict):
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
    # (the sample callbacks live next to _setup_vendor_io; they run on the
    # SDK's reader threads and only touch lock-guarded state)

    # --- RPC -------------------------------------------------------------------

    #: The runtime's symbolic api names -> the SDK client methods they invoke.
    #: The mapping IS the migration table from the vendored ``unitree_api`` ids:
    #: every call site names its intent here, the dispatcher binds it to the
    #: SDK client method, and the vendor's numeric ids stay inside the SDK.
    RPC_METHODS: dict[str, str] = {
        "loco:get_fsm_id": "get_fsm_id",
        "loco:get_fsm_mode": "get_fsm_mode",
        "loco:set_fsm_id": "set_fsm_id",
        "loco:set_velocity": "set_velocity",
        "robot_state:service_list": "service_list",
        "arm:execute_action": "execute_action",
        "arm:get_action_list": "get_action_list",
        "audio:tts": "tts_maker",
        "audio:set_rgb_led": "led_control",
        "audio:get_volume": "get_volume",
    }

    def rpc_call(
        self,
        service: str,
        api_name: str,
        parameter: Any,
        *,
        motion: bool = False,
        timeout_s: float | None = None,
        guard: Callable[[], bool] | None = None,
    ) -> vg.RpcResult:
        """One vendor RPC round trip on the SDK's typed client, serialized.

        The SDK's client objects expose no thread-safety guarantee (plain
        request/future objects over shared channels), so every call to one
        service goes through that service's dispatcher lock — callers block in
        their own callback while a call is in flight, which the callback-group
        layout below already allows for. Which calls block which thread is the
        stop contract's concern (Q7): motion calls ride the same dispatcher,
        but a stop never queues behind them because it only zeroes/latches
        local state and refuses further admission; voice/LED run on their own
        callback group and can time out without touching motion paths.

        ``api_name`` is the symbolic name from :data:`RPC_METHODS`; the
        vendor's numeric api ids never leave the SDK.

        ``timeout_s`` is a real per-call bound, not a hint: the adapter sets
        the vendor client's timeout for this one call and restores it after
        (inside the dispatcher, so two callers cannot interleave their
        windows). The bounded queries — the startup peer probe, the stop's FSM
        read, the locomotion/posture confirmation polls — depend on it: without
        it every one of them would wait out the profile's full call timeout.

        ``guard`` is the last word before the request goes out, and it runs
        **inside** the dispatcher: a caller that had to queue behind another
        vendor call re-checks there whether the command is still admitted. That
        is what keeps a stale command from being sent *after* the stop that
        superseded it — the dispatcher it waited on is the same one the stop's
        zero uses. A refused guard returns ``dropped=True`` (nothing was sent,
        so there is no unknown fate to latch).
        """
        try:
            method_name = self.RPC_METHODS[f"{service}:{api_name}"]
        except KeyError as exc:
            raise RuntimeError(f"no vendor rpc mapping for {service}:{api_name}") from exc
        client = self._rpc[service]
        method = getattr(client, method_name)
        dispatcher = self._rpc_dispatchers[service]
        # ``parameter`` is the SDK method's positional argument tuple (or None
        # for the no-arg queries); the vendor's JSON shapes live inside the
        # SDK wrappers, not at these call sites.
        args = tuple(parameter) if parameter is not None else ()
        with dispatcher, client.call_timeout(timeout_s):
            if guard is not None and not guard():
                return vg.RpcResult(ok=False, code=-1, motion=motion, api_id=api_name, dropped=True)
            outcome = method(*args)
        if isinstance(outcome, tuple) and len(outcome) == 2:
            from unitree_g1_adapter.clients import SdkCallResult  # noqa: PLC0415 - single adapter import site

            code, data = outcome
            outcome = SdkCallResult(ok=int(code) == 0, code=int(code), data=data)
        return vg.rpc_result_from_sdk(outcome, motion=motion, api_id=api_name)

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
            if target == IDLE_MODE and (self._state.stop_latched or self._arm_handover_unconfirmed):
                # A stop (or a stream exit) leaves the runtime in idle with the
                # latch engaged and, when the arm frame had weight, the
                # unacknowledged-handover warning published. Requesting idle
                # again is how the operator takes the frame back: it clears the
                # latch and the warning once the local weight is 0.
                unconfirmed = self._arm_handover_unconfirmed
                if not self._clear_stop_latch():
                    response.success = False
                    response.message = "stop still in progress or arm publication unresolved; refusing to clear latch"
                    response.valid_transitions = [IDLE_MODE]
                    return response
                response.success = True
                response.message = "stop latch cleared" + (
                    "; the remote arm handover was never acknowledged, and clearing this does not confirm it"
                    if unconfirmed
                    else ""
                )
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
            # A stream exit engages the stop latch by design (the arm frame had
            # weight), and that latch is this switch's own doing — it moves the
            # epoch the switch captured. A stop that completed after the vendor
            # work moves it too. Both leave the runtime exactly where an ``idle``
            # request wants it: idle, latched, weight target 0. Any other target
            # still needs the epoch untouched, so no mode switch can mask a stop.
            if self._state.stop_epoch != epoch and target != IDLE_MODE:
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
        if self._preset_only:
            self._state.engage_stop("HOLD")
            response.success = False
            response.cancel_latency_s = 0.0
            response.idle_latency_s = response.torque_off_latency_s = -1.0
            response.message = "admission closed locally; preset cancellation/physical stop is unverified"
            return response
        policy = str(request.policy or STOP_POLICIES[0])
        response.cancel_latency_s = -1.0
        response.idle_latency_s = -1.0
        response.torque_off_latency_s = -1.0
        if policy not in STOP_POLICIES:
            response.success = False
            response.message = f"unknown stop policy {policy!r}; supported: {list(STOP_POLICIES)}"
            return response

        with self._stop_lock:
            started = time.monotonic()
            had_stream = bool(self._latch)
            needs_base_stop = self._modes.mode == "locomotion" or self._velocity_inflight or self._locomotion_requested
            # (1) Close admission FIRST, atomically, and put no vendor call in
            #     front of it: the latch and the idle commit stop the input
            #     callbacks and the output ticks, and they outlive every vendor
            #     failure below. Every stop policy ramps the shared weight to 0,
            #     so the local target is known before any query — a vendor read
            #     that blocks or must time out can no longer delay the latch.
            with self._lock:
                self._state.engage_stop(policy)
                self._modes.commit(IDLE_MODE)
                self._twist = None
                self._twist_stamp = 0.0
                self._weight_target = projection.MODE_WEIGHT_TARGETS[IDLE_MODE]
            response.cancel_latency_s = time.monotonic() - started

            # (2) Only now ask the vendor for its own FSM report. Only the
            #     torque-releasing policy needs it: HOLD switches no FSM. A
            #     report that never arrives refuses that policy and keeps the
            #     admission closed — a failed read must never reopen it.
            fsm_id: int | None = None
            if policy == "TORQUE_OFF":
                fsm_id = self._reported_fsm_id()
                if fsm_id is None:
                    response.idle_latency_s = time.monotonic() - started
                    response.success = False
                    response.message = (
                        "the reported FSM could not be read; refusing TORQUE_OFF (fail closed); "
                        "admission is closed and the stop latch stays engaged"
                    )
                    return response
            try:
                plan = projection.resolve_stop_plan(policy, fsm_id=fsm_id, stop_config=self._vendor.get("stop"))
            except CommandRejected as rejected:
                # A balancing biped cannot always honour TORQUE_OFF. Refusing is
                # the honest answer; holding silently and reporting success is
                # not. The refusal covers the torque release only: admission is
                # already closed and stays closed.
                response.idle_latency_s = time.monotonic() - started
                response.success = False
                response.message = (
                    f"{rejected.reason}: {rejected.detail}; admission is closed and the stop latch stays engaged"
                )
                return response

            # (3) Stop walking: an explicit zero, confirmed by the vendor.
            # Q7: the stop must not queue behind a mode switch that is
            # waiting on the vendor, so the zero competes for the loco
            # dispatcher with a SHORT bounded acquisition; losing that race
            # is an honest, fail-closed outcome — admission is already closed
            # and the message says the zero is unconfirmed.
            zero = self._stop_zero_velocity() if needs_base_stop else None
            if zero is not None and not zero.ok:
                response.idle_latency_s = time.monotonic() - started
                response.success = False
                response.message = (
                    f"SetVelocity(0,0,0) {'timed out' if zero.timed_out else f'failed with code {zero.code}'}; "
                    "stop latched locally, the zero-velocity stop is unconfirmed"
                )
                return response
            if zero is not None:
                self._locomotion_requested = False

            # (4) The latched targets are already frozen (admission is closed),
            #     so what remains is handing the frame back: ramp the weight to 0.
            ramped, ramp_detail = self._await_weight_zero()
            response.idle_latency_s = -1.0 if had_stream else time.monotonic() - started
            if not ramped:
                response.success = False
                response.message = f"{ramp_detail}; stop latched locally, the weight handover is unconfirmed"
                return response
            if had_stream:
                # The same fact the exit path reports: the local frame is at
                # zero weight, the vendor's own handover is not observable.
                # Visible in status until the operator clears it with `idle`.
                self._state.add_fault(ARM_HANDOVER_WARNING)

            # (5) TORQUE_OFF releases torque through the damping FSM — never
            #     the zero-torque FSM a caller might imagine.
            if plan.fsm_id is not None:
                release = self.rpc_call("loco", "set_fsm_id", (plan.fsm_id,), motion=True)
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
        response.success = not had_stream
        response.message = (
            "local stop complete: admission closed, targets frozen, zero-weight publication complete; "
            "remote arm handover is not acknowledged by this interface"
        )
        if plan.fsm_id is not None:
            response.message += (
                f"; torque released through the damping FSM (requested {plan.downgraded_from}; "
                "the platform-safe equivalent on a balancing biped)"
            )
        return response

    def _reported_fsm_id(self) -> int | None:
        """The vendor's own FSM report, or None when it does not answer."""
        result = self.rpc_call("loco", "get_fsm_id", None, timeout_s=min(self._rpc_timeout_s, 1.0))
        # The SDK wrapper unwraps the vendor's {"data": id} to the bare id.
        if result.ok and isinstance(result.data, int):
            return result.data
        return None

    _STOP_ZERO_DISPATCH_WAIT_S = 0.25

    def _stop_zero_velocity(self) -> vg.RpcResult:
        """The stop's zero-velocity call with bounded dispatcher priority.

        A mode switch waiting on the vendor holds the loco dispatcher for up
        to one call; the stop waits at most ``_STOP_ZERO_DISPATCH_WAIT_S`` for
        it instead of the full call timeout, keeping the stop's latency
        bounded while still serializing against every other loco caller (the
        SDK clients have no thread-safety guarantee we could rely on).
        """
        dispatcher = self._rpc_dispatchers["loco"]
        if not dispatcher.acquire(timeout=self._STOP_ZERO_DISPATCH_WAIT_S):
            return vg.RpcResult(ok=False, code=-1, timed_out=True, motion=True, api_id="set_velocity")
        try:
            outcome = self._rpc["loco"].set_velocity(0.0, 0.0, 0.0, self._velocity_duration_s)
        except Exception as exc:  # noqa: BLE001 - a dispatch failure is a failed stop, not a crash
            self.get_logger().warning(f"the stop's zero-velocity dispatch failed: {exc}")
            return vg.RpcResult(ok=False, code=-1, timed_out=True, motion=True, api_id="set_velocity")
        finally:
            dispatcher.release()
        return vg.rpc_result_from_sdk(outcome, motion=True, api_id="set_velocity")

    def _clear_handover_warning(self) -> None:
        """Drop only the arm-handover warning; every other fault stays."""
        faults = self._state.faults
        remaining = [fault for fault in faults if fault != ARM_HANDOVER_WARNING]
        if len(remaining) != len(faults):
            self._state.set_status(None, remaining)

    def _clear_stop_latch(self) -> bool:
        """Clear the latch, unless a stop is still holding it.

        This is also the operator's way out of an unacknowledged arm handover.
        Once the shared weight is 0 and no publication is in flight, the local
        side of the handover is complete: an explicit ``idle`` takes the
        :data:`ARM_HANDOVER_WARNING` back and clears the latch. That is the
        operator saying "the frame is mine", not a claim that the vendor
        acknowledged the handover — and stream re-entry still runs the full
        admission (hold latch, service ownership, FSM, fresh feedback), so
        nothing about the fail-closed path gets weaker.
        """
        if self._preset_only:
            return False
        if not self._stop_lock.acquire(blocking=False):
            return False
        try:
            with self._lock:
                if self._arm_write_fault or self._arm_write_pending or self._weight > 0.0:
                    return False
                # Local handover is settled (weight 0, nothing pending): the
                # remaining unknown is the vendor's acknowledgement, which this
                # interface cannot observe and must not keep claiming.
                self._arm_handover_unconfirmed = False
                # The session is over, so drop the 17-joint frame: leaving it
                # here made "entered stream at some point" masquerade as "is in
                # stream now" (a later stop then reported success=False and
                # re-raised the warning for a stream that no longer exists).
                # The weight is 0 and no publication is pending, so no tick
                # consumes the frame any more.
                self._latch = {}
            self._clear_handover_warning()
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
        # The adapter's TtsMaker wrapper takes (text, speaker_id); the profile's
        # index stays profile data (the upstream index quirk is documented on
        # the wrapper: the vendor's own counter never leaves 0).
        result = self.rpc_call("audio", "tts", (str(text), int(tts.get("speaker_id", 0))))
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
        result = self.rpc_call("audio", "set_rgb_led", (int(request.r), int(request.g), int(request.b)))
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
            ok, detail = self._exit_stream(target_mode=mode)
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

        def guard() -> bool:
            """The zero is a motion request too: admission must still hold."""
            return self._state.stop_epoch == epoch and self._runtime_admitted()

        result = self.rpc_call(
            "loco", "set_velocity", (0.0, 0.0, 0.0, self._velocity_duration_s), motion=True, guard=guard
        )
        if self._state.stop_epoch != epoch:
            return False, "stop engaged while the mode switch was waiting on the vendor"
        if result.dropped:
            return False, "the zero-velocity frame was dropped: admission closed before it could be sent"
        if not result.ok:
            detail = "timed out; not re-sent" if result.timed_out else f"failed with vendor code {result.code}"
            return False, f"SetVelocity(0,0,0) {detail}; refusing to leave locomotion with the base still commanded"
        with self._lock:
            self._velocity_sent = (0.0, 0.0, 0.0)
            self._last_velocity_sent_s = time.monotonic()
            self._locomotion_requested = False
        return True, "locomotion left: the base velocity was zeroed and the vendor confirmed it"

    def _enter_stream(self, epoch: int) -> tuple[bool, str]:
        """Read-only admission, fresh hold latch, then arm weight ramp; no FSM writes."""
        with self._lock:
            if self._arm_write_fault or self._arm_write_pending or self._weight > 0.0 or self._arm_handover_unconfirmed:
                return False, "ARM_HANDOVER_UNKNOWN: previous stream has not completed locally"
        try:
            services = self.rpc_call("robot_state", "service_list", None)
            if not services.ok or not isinstance(services.data, list):
                return False, "ARM_OWNERSHIP_UNKNOWN: service list unavailable"
            states = {}
            for item in services.data:
                if not isinstance(item, dict) or item.get("name") in states:
                    return False, "ARM_OWNERSHIP_UNKNOWN: malformed service list"
                states[item.get("name")] = item.get("status")
            required = (self._vendor.get("arm_admission") or {}).get("stream_entry") or {}
            if not required:
                return False, (
                    "ARM_OWNERSHIP_UNKNOWN: the profile declares no vendor.arm_admission.stream_entry "
                    "service ownership, so the frame's owner cannot be established"
                )
            for name, expected in required.items():
                reported = states.get(name)
                if type(reported) is not int or reported != int(expected):
                    # 0 running / 1 stopped, straight from the profile: the same
                    # two refusals the hardcoded check produced, with the names
                    # and expectations now owned by the profile.
                    if int(expected) == 0:
                        return False, f"ARM_OWNERSHIP_UNKNOWN: {name} must already be running"
                    return False, f"ARM_OWNERSHIP_CONFLICT: {name} must already be stopped"
            fsm = self.rpc_call("loco", "get_fsm_id", None)
            allowed = (self._vendor.get("arm_sdk") or {}).get("allowed_fsm_ids", [])
            if not fsm.ok or type(fsm.data) is not int or fsm.data not in allowed:
                return False, "ARM_FSM_UNSUPPORTED: current FSM is not admitted for arm_sdk"
        except Exception as exc:  # fail closed on malformed/unavailable vendor queries
            return False, f"ARM_ADMISSION_UNAVAILABLE: {exc}"
        if self._state.stop_epoch != epoch or self._state.stop_latched:
            return False, "stop engaged while checking stream admission"
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
        with self._lock:
            if self._state.stop_epoch != epoch or self._state.stop_latched:
                return False, "stop engaged before the stream latch was committed"
            self._latch = dict(zip(projection.ARM_SDK_FRAME, hold, strict=True))
            # A fresh stream session starts with no hand targets: a target from
            # a previous session would be a motion nobody commanded.
            self._hand_latch = {"left": {}, "right": {}}
            self._weight_target = projection.MODE_WEIGHT_TARGETS["stream"]
        return True, "stream admitted: service ownership/FSM checked, hold latched; no vendor mode changed"

    def _exit_stream(self, *, target_mode: str = IDLE_MODE) -> tuple[bool, str]:
        """Ramp the shared weight back to 0 before the mode is reported inactive (D4).

        The ramp *is* the handover this platform can observe: admission closes,
        every latched target stays frozen, and the published weight reaches 0.
        Whether the vendor then hands the frame back to its own control is not
        observable through this interface, so the switch to ``idle`` SUCCEEDS
        (the public mode contract holds) while the unacknowledged remote
        handover stays visible as :data:`ARM_HANDOVER_WARNING` in
        ``RuntimeStatus``. ``idle`` is what clears that warning and the local
        latch; every other target keeps the old refusal, because only a settled
        handover may precede a new vendor mode.
        """
        with self._lock:
            self._state.engage_stop("HOLD")
            self._modes.commit(IDLE_MODE)
            self._weight_target = projection.MODE_WEIGHT_TARGETS[IDLE_MODE]
        # The latch just engaged is this exit's own generation: only a further
        # bump means a stop arrived while the ramp was running.
        latch_epoch = self._state.stop_epoch
        ramped, detail = self._await_weight_zero()
        if ramped:
            self._state.add_fault(ARM_HANDOVER_WARNING)
            if target_mode == IDLE_MODE:
                return True, (
                    "stream exited: admission closed, targets frozen, the zero-weight frame was published "
                    "locally; the remote arm handover is not acknowledged and is reported in status"
                )
            return False, "zero-weight frame published locally; remote arm handover is unconfirmed"
        if self._state.stop_epoch != latch_epoch:
            return False, "stop engaged while stream was exiting"
        return False, detail

    def _await_weight_zero(self) -> tuple[bool, str]:
        """Wait (bounded) for the 50 Hz tick to ramp the shared weight to 0."""
        with self._lock:
            weight = self._weight
            fault = self._arm_write_fault
            pending = self._arm_write_pending
        if fault:
            return False, fault
        if weight <= 0.0 and not pending:
            return True, ""
        if self._weight_ramp_down_rate <= 0.0:
            return False, "the profile's weight ramp-down rate is not positive"
        # At the stop section's rate the ramp cannot take longer than
        # weight/rate; the margin only covers timer jitter.
        deadline = time.monotonic() + weight / self._weight_ramp_down_rate + 1.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._arm_write_fault:
                    return False, self._arm_write_fault
                if self._weight <= 0.0 and not self._arm_write_pending:
                    return True, ""
            time.sleep(0.02)
        return False, "the shared weight did not ramp back to 0 within the expected time"

    def _enter_locomotion(self, epoch: int) -> tuple[bool, str]:
        """SetFsmId(start), then commit only once GetFsmId reports it (D5)."""
        with self._lock:
            self._locomotion_requested = True
        fsm_id = projection.FSM_IDS["start"]
        result = self.rpc_call("loco", "set_fsm_id", (fsm_id,), motion=True)
        if self._state.stop_epoch != epoch:
            return False, "stop engaged while the mode switch was waiting on the vendor"
        if not result.ok:
            detail = "timed out; not re-sent" if result.timed_out else f"failed with vendor code {result.code}"
            return False, f"SetFsmId({fsm_id}) {detail}"
        deadline = time.monotonic() + self._rpc_timeout_s
        while time.monotonic() < deadline:
            if self._state.stop_epoch != epoch:
                return False, "stop engaged while waiting for the vendor to confirm the FSM"
            answer = self.rpc_call("loco", "get_fsm_id", None, timeout_s=0.5)
            # The SDK wrapper unwraps the vendor's {"data": id} JSON to the
            # bare id (adapter-verified behavior).
            if answer.ok and answer.data == fsm_id:
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
            if self._arm_write_fault:
                return
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
        frame = projection.arm_sdk_lowcmd(
            latch,
            weight=new_weight,
            mode_pr=self._mode_pr,
            mode_machine=self._mode_machine_expected,
            kp=self._arm_sdk_kp,
            kd=self._arm_sdk_kd,
        )
        # The adapter's CommandWriter accepts the projection's frame mapping
        # (slot list + the frame-level mode_pr/mode_machine/crc) and builds
        # the vendor LowCmd_ on the SDK path; a refused write is counted, not
        # assumed delivered.
        with self._lock:
            # Recheck after frame construction: a stop must discard unsent ramp-up.
            if new_weight > weight and not (self._runtime_admitted() and self._modes.spec().allows_stream):
                return
            self._arm_write_pending = True
            if new_weight > 0.0:
                self._arm_handover_unconfirmed = True
        try:
            sent = self._arm_sdk_writer.write(frame)
            detail = "writer refused the frame"
        except Exception as exc:  # publication outcome unknown; never auto-retry
            sent, detail = False, str(exc)
        with self._lock:
            self._arm_write_pending = False
            if sent:
                self._weight = new_weight
            else:
                self._arm_write_fault = f"ARM_HANDOVER_UNKNOWN: {detail}; no automatic retry"
                self._weight_target = 0.0
        if not sent:
            self._state.engage_stop("HOLD")
            self._modes.commit(IDLE_MODE)
            self.get_logger().error(self._arm_write_fault)

    def _dex3_tick(self) -> None:
        """Publish latched hand targets at the profile's Dex3 rate, in stream mode."""
        if not (self._modes.spec().allows_stream and self._runtime_admitted()):
            return
        with self._lock:
            targets = {side: dict(latch) for side, latch in self._hand_latch.items()}
        for side, latched in targets.items():
            if not latched:
                continue  # a hand nobody commanded is not commanded at all
            frame = projection.dex3_handcmd(latched, side=side, kp=self._dex3_kp, kd=self._dex3_kd)
            if not self._hand_writers[side].write(frame["motor_cmd"]):
                self.get_logger().warning(
                    f"the vendor refused the {side} hand command frame (write failed)",
                    throttle_duration_sec=5.0,
                )

    def _velocity_tick(self) -> None:
        """Issue SetVelocity for the current command at the profile's duration.

        A command that differs from the last one sent goes out at once —
        waiting out the previous command's duration would delay a stop or a
        change of direction by up to that duration. The duration throttle
        applies only to repeating the SAME command, which is the renewal the
        profile describes.

        The SDK's ``SetVelocity`` blocks up to the client timeout on the
        calling thread, so the tick does NOT call it directly: a 5 s stall
        would freeze the 50 Hz tick group the stop ramp depends on. Instead
        the tick hands the command to a single-flight worker thread
        (:meth:`_velocity_worker`) and only records the dispatch. The worker
        serializes through the loco dispatcher like every other loco call; a
        command dispatched while an earlier one is still in flight is the
        caller's (this tick's) problem and is dropped by the worker — the
        latest command is always re-dispatched by the next tick, and a motion
        request whose fate is unknown is never re-sent (Q7): a failed or
        timed-out dispatch drops the cached command entirely, and only a
        fresh upstream Twist re-arms the cadence.

        Admission is checked twice: here (cheap, before a thread is worth
        starting) and again in the worker, under the loco dispatcher, against
        the ``(stop generation, mode)`` snapshot this tick records. A command
        that had to wait behind the stop's zero must not be sent after it.
        """
        with self._lock:
            inflight = self._velocity_inflight
            twist = self._twist
            twist_stamp = self._twist_stamp
            last_sent = self._last_velocity_sent_s
            sent = self._velocity_sent
        now = time.monotonic()
        if inflight:
            return  # one call at a time; the latest command is already cached or will re-tick
        if not (self._modes.spec().allows_base and self._runtime_admitted()):
            return
        if twist is None or now - twist_stamp > self._twist_staleness_s:
            return
        if twist == sent and now - last_sent < self._velocity_duration_s:
            return  # the same command is still within its issued duration
        payload = (twist[0], twist[1], twist[2], self._velocity_duration_s)
        # The generation this command belongs to, captured where admission was
        # granted: the worker may have to queue behind the loco dispatcher, and
        # a command accepted before a stop must never be sent after it.
        snapshot = (self._state.stop_epoch, self._modes.mode)
        with self._lock:
            self._velocity_inflight = True
            self._velocity_sent = twist
            self._last_velocity_sent_s = now
        thread = threading.Thread(target=self._velocity_worker, args=(payload, snapshot), daemon=True)
        thread.start()

    def _velocity_worker(self, payload: tuple[float, float, float, float], snapshot: tuple[int, str]) -> None:
        """The single-flight SetVelocity executor; one thread per dispatch.

        ``snapshot`` is the (stop generation, mode) pair the command was admitted
        under. The guard re-checks it under the loco dispatcher, so a command
        that waited behind the stop's zero-velocity call is dropped instead of
        being sent afterwards — otherwise the base would keep walking for up to
        ``command_duration_s`` on a command nobody admits any more.
        """
        epoch, mode = snapshot

        def guard() -> bool:
            return (
                self._state.stop_epoch == epoch
                and self._modes.mode == mode
                and self._runtime_admitted()
                and self._modes.spec().allows_base
            )

        try:
            result = self.rpc_call("loco", "set_velocity", payload, motion=True, guard=guard)
        except Exception as exc:  # noqa: BLE001 - a dispatch must never kill the runtime
            self.get_logger().warning(f"SetVelocity dispatch failed: {exc}")
            result = None
        finally:
            with self._lock:
                self._velocity_inflight = False
        if result is not None and not result.ok:
            # A command that was never sent, or whose fate is unknown, is not
            # re-sent: the cached command is abandoned and only a fresh upstream
            # Twist re-arms it.
            with self._lock:
                self._twist = None
                self._twist_stamp = 0.0
                self._velocity_sent = None
            if result.dropped:
                self.get_logger().info(
                    "SetVelocity not sent: admission closed or the mode changed while it waited for the vendor"
                )
            else:
                self.get_logger().warning(
                    f"SetVelocity {'timed out' if result.timed_out else f'failed with code {result.code}'}; "
                    "the cached velocity command was dropped, a fresh Twist re-arms it"
                )

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

    def admit_preset_attempt(self) -> tuple[bool, str]:
        """Read-only preflight for each independent request; dispatch is serialized."""
        with self._lock:
            if self._state.stop_latched:
                return False, "preset stop latched or prior outcome unknown; no dispatch"
        if not self._preset_physical:
            return True, "mock transport"
        try:
            services = self.rpc_call("robot_state", "service_list", None)
            if not services.ok:
                failure = "timed out" if services.timed_out else "failed"
                return False, (
                    f"service ownership query failed: robot_state.service_list {failure} "
                    f"(SDK code {services.code}); preset not dispatched"
                )
            if not isinstance(services.data, list):
                return False, (
                    "service ownership query failed: robot_state.service_list returned an invalid response "
                    "(expected a service list); preset not dispatched"
                )
            required = (self._vendor.get("preset_commissioning") or {}).get("required_services") or {}
            if not required:
                return False, (
                    "PRESET_ADMISSION_UNCONFIGURED: the profile declares no "
                    "vendor.preset_commissioning.required_services ownership check"
                )
            for name, expected in required.items():
                matches = [s for s in services.data if isinstance(s, dict) and s.get("name") == name]
                if (
                    len(matches) != 1
                    or type(matches[0].get("status")) is not int
                    or matches[0]["status"] != int(expected)
                ):
                    return False, f"{name} is not confirmed running"
            fsm = self.rpc_call("loco", "get_fsm_id", None)
            expected = (self._vendor.get("operator_validation") or {}).get("expected_fsm_id")
            if not fsm.ok or type(expected) is not int or type(fsm.data) is not int or fsm.data != expected:
                return False, "FSM changed or unavailable; no automatic mode switch"
            with self._lock:
                if not projection.is_fresh(self._lowstate_stamp, time.monotonic(), self._feedback_window_s):
                    return False, "joint feedback is stale"
                if self._state.stop_latched:
                    return False, "stop engaged during preflight"
            return True, f"operator-supervised preset attempt in observed FSM {fsm.data}"
        except Exception as exc:
            return False, f"preset preflight failed: {exc}"

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
        # Release the adapter's channels; the vendor factory itself is a
        # process-wide singleton with no teardown API (its reader threads
        # are daemons), so closing the feeds/writers/clients is the full
        # cleanup the SDK offers. Teardown must not mask the exit path.
        with contextlib.suppress(Exception):
            node._sdk.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
