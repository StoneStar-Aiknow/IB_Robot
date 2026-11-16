"""The normalized port between the simulation core and one platform adapter.

A platform adapter translates a simulator's native observation and action formats into
these objects and back; the core never sees a native key, an arm count, a DOF or a tick
length that the adapter's ``AdapterDescriptor`` did not declare. Units are metres and
radians; poses are ``[x, y, z, qw, qx, qy, qz]`` in the descriptor's ``pose_frame``.

Native scores, rewards and success flags never travel in an ``ObservationFrame``: the only
carrier is ``PlatformResult``, which the engine hands to its result listener and nowhere else.

A profile keeps the settings of this layer in its platform section, ``simulation.<platform>``
(the same place the registry's platforms keep theirs); ``platform_settings`` reads it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

SCHEMA_VERSION = 1

CLOCK_PLATFORM_PULL = "platform_pull"  # the platform owns the loop and asks for each tick
CLOCK_HOST_STEP = "host_step"  # the runtime calls the platform's step itself
CLOCK_MODES = frozenset({CLOCK_PLATFORM_PULL, CLOCK_HOST_STEP})

COMMAND_JOINT = "joint"
COMMAND_GRIPPER = "gripper"
COMMAND_POSE = "pose"
COMMAND_IK = "ik"
COMMAND_KINDS = frozenset({COMMAND_JOINT, COMMAND_GRIPPER, COMMAND_POSE, COMMAND_IK})

IK_NONE = "none"
IK_PLATFORM = "platform"  # IK requests travel to the platform with a tick and come back with the next one


class DescriptorError(ValueError):
    """An adapter descriptor or profile is inconsistent; the runtime refuses to start."""


@dataclass(frozen=True)
class GroupSpec:
    """One motion group (an arm): its joints in command order and an optional 1-D gripper."""

    name: str
    joints: tuple[str, ...]
    gripper: str | None = None
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_wrap_limits: tuple[float, ...] = ()

    @property
    def dof(self) -> int:
        return len(self.joints)


@dataclass(frozen=True)
class AdapterDescriptor:
    platform: str
    clock_mode: str
    dt_s: float
    groups: tuple[GroupSpec, ...]
    supported_commands: frozenset[str]
    ik: str = IK_NONE
    # Orientation error the platform IK guarantees for a solution it reports (rad). A request
    # asking for a tighter tolerance is refused; 0 = not declared (only default requests).
    ik_orientation_tolerance_rad: float = 0.0
    # Whether a tick already submitted to the platform can be withdrawn. No adapter can undo
    # executed physics; a False here means cancel takes effect from the next tick only.
    atomic_tick_cancellable: bool = False
    pose_frame: str = "world"
    cameras: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise DescriptorError(f"adapter descriptor schema {self.schema_version} != {SCHEMA_VERSION}")
        if self.clock_mode not in CLOCK_MODES:
            raise DescriptorError(f"clock_mode {self.clock_mode!r} not in {sorted(CLOCK_MODES)}")
        if not (math.isfinite(self.dt_s) and self.dt_s > 0.0):
            raise DescriptorError(f"dt_s must be a finite positive number, got {self.dt_s!r}")
        if not str(self.pose_frame).strip():
            raise DescriptorError("pose_frame must name the frame every pose is expressed in")
        if not math.isfinite(self.ik_orientation_tolerance_rad) or self.ik_orientation_tolerance_rad < 0.0:
            raise DescriptorError("ik_orientation_tolerance_rad must be a finite number >= 0")
        if not self.groups:
            raise DescriptorError("an adapter declares at least one motion group")
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            raise DescriptorError(f"duplicate motion groups {names}")
        for group in self.groups:
            if group.dof <= 0:
                raise DescriptorError(f"group {group.name!r} has no joints")
            if len(group.tcp_offset) != 3 or not all(math.isfinite(v) for v in group.tcp_offset):
                raise DescriptorError(f"group {group.name!r} tcp_offset must be 3 finite numbers (m)")
            if group.joint_wrap_limits and (
                len(group.joint_wrap_limits) != group.dof
                or not all(math.isfinite(v) and v > 0 for v in group.joint_wrap_limits)
            ):
                raise DescriptorError(f"group {group.name!r} joint_wrap_limits must give one positive limit per joint")
        unknown = set(self.supported_commands) - COMMAND_KINDS
        if unknown:
            raise DescriptorError(f"unknown command kinds {sorted(unknown)}")
        if self.ik not in (IK_NONE, IK_PLATFORM):
            raise DescriptorError(f"ik {self.ik!r} must be {IK_NONE!r} or {IK_PLATFORM!r}")
        if self.ik == IK_NONE and {COMMAND_POSE, COMMAND_IK} & set(self.supported_commands):
            raise DescriptorError("pose / ik commands need an IK capability; this adapter declares none")

    def group(self, name: str) -> GroupSpec:
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(name)

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.groups)

    def supports(self, kind: str) -> bool:
        return kind in self.supported_commands


@dataclass(frozen=True)
class ObservationFrame:
    """One normalized observation after a platform tick (or at session start)."""

    session_id: str
    generation: int
    step_id: int  # platform control ticks executed in this session
    joints: Mapping[str, Sequence[float]]  # group -> joint positions, command order
    grippers: Mapping[str, float] = field(default_factory=dict)  # group -> 0 closed .. 1 open
    ee_poses: Mapping[str, Sequence[float]] = field(default_factory=dict)  # group -> flange pose
    images: Mapping[str, Any] = field(default_factory=dict)  # camera -> HxWx3 uint8
    wall_time: float = 0.0


@dataclass(frozen=True)
class IkRequest:
    request_id: int
    group: str
    pose: tuple[float, ...]  # flange pose (TCP offset already removed)


@dataclass(frozen=True)
class IkReply:
    request_id: int
    ok: bool
    joints: tuple[float, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class TickDecision:
    """What the core wants for the next tick.

    ``targets is None`` means idle: the platform must not step physics. A decision with
    ``ik_requests`` but no targets also does not step.
    """

    targets: Mapping[str, Mapping[str, Any]] | None
    ik_requests: tuple[IkRequest, ...] = ()
    command_id: str | None = None
    command_seq: int | None = None  # per-runtime counter of the command, for tick ledgers
    kind: str = "idle"


@dataclass(frozen=True)
class PlatformResult:
    """The platform's own end-of-session result (a verdict, when the platform judges one).

    The engine hands it to its result listener only; it is never part of an observation.
    """

    native_success: bool | None
    native_score: float | None = None
    terminated: bool = True
    truncated: bool = False
    native_reason: str | None = None
    take_action_cnt: int | None = None
    step_lim: int | None = None
    protocol_version: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)


class PlatformCodec(Protocol):
    """What every adapter provides on top of its transport."""

    descriptor: AdapterDescriptor

    def decode_observation(
        self, native: Mapping[str, Any], *, session_id: str, generation: int
    ) -> ObservationFrame: ...

    def encode_action(self, targets: Mapping[str, Mapping[str, Any]]) -> Any: ...


def platform_settings(profile: Mapping[str, Any], platform: str) -> dict[str, Any]:
    """The profile's ``simulation.<platform>`` section (empty when the profile has none)."""
    section = (profile.get("simulation") or {}).get(platform) or {}
    if not isinstance(section, Mapping):
        raise DescriptorError(f"runtime profile simulation.{platform} must be a mapping")
    return dict(section)


def descriptor_from_profile(profile: Mapping[str, Any], *, platform: str, clock_mode: str) -> AdapterDescriptor:
    """Build a descriptor from a runtime profile's ``motion_groups`` and ``simulation.<platform>``.

    The runtime profile (robot_runtime spec) is the single source of joints, groups, TCP and
    cameras; adapters only add what their platform protocol can do.
    """
    sim = platform_settings(profile, platform)
    groups = []
    for name, spec in dict(profile.get("motion_groups") or {}).items():
        spec = dict(spec or {})
        groups.append(
            GroupSpec(
                name=str(name),
                joints=tuple(str(j) for j in spec.get("joints") or ()),
                gripper=str(spec["gripper"]) if spec.get("gripper") else None,
                tcp_offset=tuple(float(v) for v in spec.get("tcp_offset", (0.0, 0.0, 0.0))),
                joint_wrap_limits=tuple(float(v) for v in spec.get("joint_wrap_limits", ())),
            )
        )
    if "dt_s" not in sim:
        raise DescriptorError(f"runtime profile simulation.{platform}.dt_s is required")
    return AdapterDescriptor(
        platform=platform,
        clock_mode=str(sim.get("clock_mode", clock_mode)),
        dt_s=float(sim["dt_s"]),
        groups=tuple(groups),
        supported_commands=frozenset(str(c) for c in sim.get("supported_commands", ())),
        ik=str(sim.get("ik", IK_NONE)),
        ik_orientation_tolerance_rad=float(sim.get("ik_orientation_tolerance_rad", 0.0)),
        atomic_tick_cancellable=bool(sim.get("atomic_tick_cancellable", False)),
        pose_frame=str(profile.get("pose_frame", "")),
        cameras=tuple(str(c) for c in profile.get("cameras", ())),
    )


def profile_mismatches(profile: Mapping[str, Any], descriptor: AdapterDescriptor) -> list[str]:
    """Where the capabilities a runtime profile publishes disagree with what the adapter does.

    The profile's capabilities go out in RuntimeStatus and the interface description; the
    descriptor decides which services exist. A runtime refuses to start on any mismatch.
    """
    caps = dict(profile.get("capabilities") or {})
    problems = []

    def pair(capability: str, kind: str) -> None:
        if (capability in caps) != descriptor.supports(kind):
            problems.append(
                f"capability {capability!r} {'declared' if capability in caps else 'missing'} but the adapter "
                f"{'does not support' if capability in caps else 'supports'} {kind!r} commands"
            )

    pair("motion.move_to_pose", COMMAND_POSE)
    pair("motion.ik", COMMAND_IK)
    move = caps.get("motion.move_to_pose") or {}
    if move.get("groups") and tuple(move["groups"]) != descriptor.group_names:
        problems.append(
            f"motion.move_to_pose groups {list(move['groups'])} != adapter groups {list(descriptor.group_names)}"
        )
    grippers = sum(1 for g in descriptor.groups if g.gripper)
    count = (caps.get("gripper.1d") or {}).get("count")
    if count is not None and descriptor.supports(COMMAND_GRIPPER) and int(count) != grippers:
        problems.append(f"gripper.1d.count {count} != adapter grippers {grippers}")
    sim = platform_settings(profile, descriptor.platform)
    if sim.get("clock_mode") and sim["clock_mode"] != descriptor.clock_mode:
        problems.append(
            f"simulation.{descriptor.platform}.clock_mode {sim['clock_mode']!r} "
            f"!= adapter clock mode {descriptor.clock_mode!r}"
        )
    return problems


def ik_request_refusal(
    descriptor: AdapterDescriptor,
    *,
    frame_id: str,
    seed_names: Sequence[str],
    orientation_tolerance: float,
    timeout_s: float,
) -> str:
    """Why a ComputeIk request cannot be served as asked ("" = it can). Nothing is approximated."""
    if frame_id and frame_id != descriptor.pose_frame:
        return f"target frame {frame_id!r} is not the runtime pose frame {descriptor.pose_frame!r}; no transform here"
    if seed_names:
        return "this platform's IK takes no seed state"
    if not math.isfinite(orientation_tolerance) or orientation_tolerance < 0:
        return "orientation_tolerance must be a finite number >= 0"
    if orientation_tolerance and orientation_tolerance < descriptor.ik_orientation_tolerance_rad - 1e-12:
        return (
            f"orientation_tolerance {orientation_tolerance} is tighter than the platform IK guarantees "
            f"({descriptor.ik_orientation_tolerance_rad})"
        )
    if orientation_tolerance and not descriptor.ik_orientation_tolerance_rad:
        return "the platform IK declares no orientation tolerance; only the default (0) is served"
    if not math.isfinite(timeout_s) or timeout_s < 0:
        return "timeout must be a finite number >= 0"
    return ""
