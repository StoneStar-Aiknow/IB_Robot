"""The X2 runtime's data plane: what ``aimdk_dataplane`` runs, and how the runtime talks to it.

The runtime is split in two processes. ``aimdk_dataplane`` (C++) takes every
joint, hand and arm command sample the HAL publishes, publishes the public joint
state views, hand touch and arm command echo, and owns the fixed-rate command
stream to the motion-control unit, public command channels in. This process
keeps the control plane: modes, arbitration, stop, the vendor services and
everything else that runs at human rates.

Why: one Python process converting ~2800 vendor samples a second under one GIL
held the X2's 310P robot-side host to /joint_states at 0.5 Hz against 50
(2026-10-26), and every way of reading fewer of them was a per-deployment patch
that left the runtime blind to a fault code present in one frame. The same
intake in C++ takes every sample at ~0.35 of a core on that host.

The two sides share three internal topics: the runtime's control word (what is
admitted, latched), the data plane's status (feedback ages, faults, rejections,
the measured arm) and the named motion's arm targets. Everything static comes
from the profile through :func:`plan`, so the profile stays the one source.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aimdk_robot import projection

CONTROL_TOPIC = "/aimdk_runtime/dataplane/control"
STATUS_TOPIC = "/aimdk_runtime/dataplane/status"
NAMED_ARM_TOPIC = "/aimdk_runtime/dataplane/named_arm"

#: Which vendor hand sub-mode a hand command channel speaks. The sub-mode
#: follows the channel the command arrived on, so a deployment can offer joint
#: control and gestures without a mode switch between them.
CHANNEL_HAND_TYPES = {
    "gripper_stream": "claw",
    "hand_stream": "dexterous_joint",
    "hand_gesture_stream": "dexterous_gesture",
}

#: The arm command echo interface: what the motion-control unit is driving the arm to.
ARM_COMMAND_STATE = "joint.arm_command_state"

#: How long the data plane keeps sending on the runtime's last word. Past it the
#: runtime is taken to be gone or stalled: cached commands are dropped and a zero
#: velocity is sent, so a dead control plane never leaves a target streaming.
CONTROL_TIMEOUT_S = 0.5

#: How often the data plane reports back.
STATUS_RATE_HZ = 20.0


def joint_state_views(profile: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """``(endpoint, joint order)`` of every published JointState interface the profile declares."""
    views = []
    for interface in (profile.get("interfaces") or {}).values():
        if (
            interface.get("kind") == "topic"
            and interface.get("direction") == "publish"
            and interface.get("message_type") == "sensor_msgs/msg/JointState"
            and interface.get("joint_names")
        ):
            views.append((str(interface["endpoint"]), [str(joint) for joint in interface["joint_names"]]))
    return views


def arm_command_state(profile: dict[str, Any]) -> tuple[str, list[str]] | None:
    """``(endpoint, joint order)`` of the arm command echo the profile declares, or None."""
    interface = (profile.get("interfaces") or {}).get(ARM_COMMAND_STATE)
    if not interface or not interface.get("joint_names"):
        return None
    return str(interface["endpoint"]), [str(joint) for joint in interface["joint_names"]]


def _interface_rate(profile: dict[str, Any], endpoint: str, default: float) -> float:
    for interface in (profile.get("interfaces") or {}).values():
        if str(interface.get("endpoint", "")) == endpoint and interface.get("rate_hz"):
            return float(interface["rate_hz"])
    return default


def plan(profile: dict[str, Any]) -> dict[str, Any]:
    """Everything static ``aimdk_dataplane`` needs, resolved from the runtime profile."""
    vendor = profile.get("vendor") or {}
    topics = vendor.get("topics") or {}
    qos = vendor.get("qos")
    command = vendor.get("command") or {}
    feedback = vendor.get("feedback") or {}
    hand = vendor.get("hand") or {}
    rate_hz = projection.joint_state_rate_hz(profile)

    def endpoint(key: str, topic: Any) -> dict[str, Any]:
        reliability, durability, depth = projection.vendor_qos(qos, key)
        return {"topic": str(topic), "reliability": reliability, "durability": durability, "depth": depth}

    echo = arm_command_state(profile)
    hand_state = endpoint("hand_state", topics["hand_state"]) if topics.get("hand_state") else None
    touch_topic = str(hand.get("touch_topic", "")) if hand_state else ""
    locomotion = projection.LocomotionLimits.from_profile(vendor.get("locomotion"))
    capabilities = profile.get("capabilities") or {}
    return {
        "joint_state_rate_hz": rate_hz,
        "command_rate_hz": float(command.get("rate_hz", 50.0)),
        "status_rate_hz": STATUS_RATE_HZ,
        "feedback_window_s": float(feedback.get("staleness_s", 0.5)),
        "stream_staleness_s": float(command.get("stream_staleness_s", 0.5)),
        "cmd_vel_staleness_s": float((capabilities.get("base.cmd_vel") or {}).get("staleness_s", 0.5)),
        "control_timeout_s": CONTROL_TIMEOUT_S,
        "source": str((vendor.get("input_source") or {}).get("name", "ibrobot.aimdk_robot")),
        "joint_state": [
            {"group": str(group), **endpoint("joint_state", topic)}
            for group, topic in (topics.get("joint_state") or {}).items()
        ],
        "hand_state": hand_state,
        "arm_command": endpoint("arm_command", topics["arm_command"]) if echo and topics.get("arm_command") else None,
        "upper_body_command": endpoint("upper_body_command", topics.get("upper_body_command", "")),
        "locomotion_command": endpoint(
            "locomotion_command", topics.get("locomotion_command", "/aima/mc/locomotion/velocity")
        ),
        "views": [
            {"topic": str(profile["joint_state_topic"]), "order": [str(joint) for joint in profile["joints"]]},
            *({"topic": topic, "order": order} for topic, order in joint_state_views(profile)),
        ],
        "touch": {"topic": touch_topic} if touch_topic else None,
        "touch_rate_hz": _interface_rate(profile, touch_topic, rate_hz),
        "arm_command_state": {"topic": echo[0], "order": echo[1]} if echo else None,
        "arm_command_state_rate_hz": _interface_rate(profile, echo[0], rate_hz) if echo else rate_hz,
        "channels": [
            {
                "name": str(channel["channel"]),
                "topic": str(channel["topic"]),
                "twist": str(channel.get("type", "float64_array")).lower() == "twist",
                "joints": [str(joint) for joint in channel.get("joints", [])],
                "modes": [str(mode) for mode in channel.get("modes", [])],
                "hand_type": CHANNEL_HAND_TYPES.get(str(channel["channel"]), ""),
            }
            for channel in profile.get("command_channels", [])
        ],
        "head_limits": {
            str(joint): float(limit) for joint, limit in ((vendor.get("head") or {}).get("limits") or {}).items()
        },
        "hand": {
            "command_min": float(hand.get("command_min", 0.0)),
            "command_max": float(hand.get("command_max", 1.0)),
            "gesture_ids": [int(value) for value in hand["gesture_ids"]] if hand.get("gesture_ids") else None,
            "default_sub_mode": int(hand.get("sub_mode", projection.HAND_SUB_MODE_CLAW)),
        },
        "locomotion": {
            "deadband": locomotion.deadband,
            "min_linear_mps": locomotion.min_linear_mps,
            "min_angular_radps": locomotion.min_angular_radps,
            "max_linear_mps": locomotion.max_linear_mps,
            "max_angular_radps": locomotion.max_angular_radps,
        },
        "mc_actions": {str(mode): str(action) for mode, action in (vendor.get("mc_actions") or {}).items()},
        "navigation_gate": "base.navigation_gate" in capabilities,
        "control_topic": CONTROL_TOPIC,
        "status_topic": STATUS_TOPIC,
        "named_arm_topic": NAMED_ARM_TOPIC,
    }


@dataclass
class Control:
    """The runtime's word to the data plane. ``clear`` and ``safe`` are counters: moving one asks once."""

    seq: int = 0
    clear: int = 0
    safe: int = 0
    mode: str = ""
    allows_stream: bool = False
    admitted: bool = False
    navigation_enabled: bool = False
    arm_owner: int | None = None
    envelope: tuple[float, float] | None = None

    def encode(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "clear": self.clear,
                "safe": self.safe,
                "mode": self.mode,
                "allows_stream": self.allows_stream,
                "admitted": self.admitted,
                "navigation_enabled": self.navigation_enabled,
                "arm_owner": self.arm_owner,
                "envelope": list(self.envelope) if self.envelope else None,
            }
        )


@dataclass
class Status:
    """What the data plane reported last.

    ``at`` is when the data plane generated the report, on this host's
    monotonic clock (the one ``time.monotonic()`` reads); the ages are seconds
    at that moment, so a consumer adds the time since ``at`` to get an
    end-to-end age — the time the report spent queued before this process saw
    it counts, a report that sat past the window is not "just generated".
    """

    control_seq: int = 0
    at: float = float("-inf")
    group_ages: dict[str, float] = field(default_factory=dict)
    hand_types: list[int] = field(default_factory=list)
    arm_measured: list[float] | None = None
    #: How old the arm measurement's stalest contributing group was at ``at``.
    arm_measured_age_s: float | None = None
    #: Per group that reported since the previous status: the joints with a non-zero code.
    faults: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)
    public_arm_count: int = 0
    public_arm_age_s: float | None = None
    #: The arm owner the data plane applied; a claim it refused stays ``None``.
    arm_owner: int | None = None

    @classmethod
    def decode(cls, text: str) -> Status:
        data = json.loads(text)
        return cls(
            control_seq=int(data.get("control_seq", 0)),
            at=float(data["at"]),
            group_ages={str(group): float(age) for group, age in (data.get("groups") or {}).items()},
            hand_types=[int(value) for value in data.get("hand_types") or []],
            arm_measured=[float(value) for value in data["arm_measured"]] if data.get("arm_measured") else None,
            arm_measured_age_s=(None if data.get("arm_measured_age_s") is None else float(data["arm_measured_age_s"])),
            faults={
                str(group): [(str(name), int(code)) for name, code in codes]
                for group, codes in (data.get("faults") or {}).items()
            },
            rejected=[str(note) for note in data.get("rejected") or []],
            public_arm_count=int(data.get("public_arm_count", 0)),
            public_arm_age_s=None if data.get("public_arm_age_s") is None else float(data["public_arm_age_s"]),
            arm_owner=None if data.get("arm_owner") is None else int(data["arm_owner"]),
        )


def start_process(profile: dict[str, Any]) -> subprocess.Popen:
    """Run ``aimdk_dataplane`` for ``profile`` outside a launch, as runtime.launch.py would.

    For tests and tools that run the runtime node in-process against the vendor
    mock: the data plane is still its own process there, as deployed.
    """
    from ament_index_python.packages import get_package_prefix  # noqa: PLC0415

    executable = Path(get_package_prefix("aimdk_dataplane")) / "lib" / "aimdk_dataplane" / "aimdk_dataplane"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="aimdk_dataplane_", delete=False) as parameters:
        json.dump({"aimdk_dataplane": {"ros__parameters": {"plan_json": json.dumps(plan(profile))}}}, parameters)
    return subprocess.Popen([str(executable), "--ros-args", "--params-file", parameters.name])
