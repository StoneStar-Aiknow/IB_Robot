"""Safe-stop core for the scheduled action dispatcher.

Builds the resolved safe-stop channel mapping from
the action contract + robot joint order, validate joint observations, and
construct `zeros`/`hold` safety commands per channel. The legacy dispatcher
intentionally keeps its original stop behavior so the disabled scheduler path
remains backward-compatible.

Design contract:
  - `zeros` channel: zeroed.
  - `hold` channel: prefer the last published target; only fall back to the
    latest validated joint value for channels whose selector uniquely maps to
    a JointState position. A hold channel with neither a last action nor a
    valid observation must NOT fabricate a value -> safe_stop() fails fatal.
  - The "complete command" is per-channel validated before publish; channels
    are published zeros-first, then hold. Any publish/serialization failure
    returns failure; already-published safety commands are NOT rolled back.
  - Only the most recent complete sample within the freshness policy updates
    the joint snapshot. Invalid/stale/unmappable samples are dropped.
  - The joint snapshot, last action, queue and smoother share one state lock.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class SafeStopError(Exception):
    """Raised when a safety command cannot be constructed (fatal)."""


@dataclass(frozen=True)
class SafeStopChannel:
    """One action contract channel resolved for safe-stop.

    A channel is a single publish target (one action spec). Each channel has an
    ordered list of positions; `hold` channels may resolve each position to a
    joint index in the JointState (1:1) for fallback-to-observation.
    """

    spec_index: int
    topic: str
    ros_type: str
    safety_behavior: str  # "zeros" | "hold"
    channel_names: list[str]  # selector.names, positional order
    joint_indices: list[int | None]  # index into joint order, None if unmappable
    clamp: tuple[float, float] | None


@dataclass(frozen=True)
class SafeStopPlan:
    """Fully-resolved safe-stop plan for one contract."""

    channels: list[SafeStopChannel]
    # fingerprint input: derived so it can be folded into the compatibility
    # fingerprint. The actual hash is computed by the caller.
    joint_order: list[str]

    @property
    def total_positions(self) -> int:
        return sum(len(c.channel_names) for c in self.channels)


@dataclass
class JointSnapshot:
    """Latest validated joint observation sample.

    Stored under the same state lock as last-action/queue. Structural validity
    is established by ``validate_joint_state``; freshness is measured from the
    local monotonic receive time assigned by the ROS adapter.
    """

    positions: list[float] = field(default_factory=list)
    received_monotonic_ns: int = 0
    valid: bool = False


def build_safe_stop_plan(
    *,
    action_specs: Sequence,
    joint_order: Sequence[str],
    name_index_of: Callable[[str, Sequence[str]], int | None] | None = None,
) -> SafeStopPlan:
    """Derive the resolved safe-stop channel mapping.

    `action_specs` is a sequence of objects with `.safety_behavior`, `.names`
    (or `.selector["names"]`), `.ros_type`/`.topic`, and `.clamp` — the
    SpecView shape from robot_config.contract_utils is the intended input.

    `joint_order` is the robot's canonical joint name order. A hold channel's
    `action.N` name is mapped to a joint index iff there is exactly one
    positional way to map N -> joint index; otherwise the position is
    unmappable (None) and must use the last-action target only.
    """
    channels: list[SafeStopChannel] = []
    joint_list = list(joint_order)
    lookup = name_index_of or _contract_name_index_of
    for spec_index, spec in enumerate(action_specs):
        names = _spec_names(spec)
        behavior = _spec_safety_behavior(spec).lower()
        if behavior not in ("zeros", "hold"):
            raise SafeStopError(f"channel {spec_index}: unknown safety_behavior {behavior!r}")
        # map each channel name to a joint index. For hold channels, prefer a
        # unique numeric suffix match to the joint order; zeros channels need
        # no observation fallback (None everywhere).
        joint_indices: list[int | None] = []
        for name in names:
            if behavior == "zeros":
                joint_indices.append(None)
            else:
                joint_indices.append(lookup(name, joint_list))
        channels.append(
            SafeStopChannel(
                spec_index=spec_index,
                topic=_spec_topic(spec),
                ros_type=_spec_ros_type(spec),
                safety_behavior=behavior,
                channel_names=names,
                joint_indices=joint_indices,
                clamp=_spec_clamp(spec),
            )
        )
    return SafeStopPlan(channels=channels, joint_order=joint_list)


def validate_joint_state(
    *,
    joint_names: Sequence[str],
    positions: Sequence[float],
    expected_joint_order: Sequence[str],
) -> JointSnapshot:
    """Validate a JointState sample: name set, uniqueness, position
    length, finite values; reorder into the resolved safe-stop channel order.

    Returns a JointSnapshot (valid=False if rejected). The caller records the
    ROS source stamp and local monotonic receive time on the snapshot.
    """
    expected = list(expected_joint_order)
    expected_set = set(expected)
    names = list(joint_names)
    if len(names) != len(expected) or set(names) != expected_set:
        return JointSnapshot(valid=False)
    if len(positions) != len(expected):
        return JointSnapshot(valid=False)
    if not all(_is_finite(p) for p in positions):
        return JointSnapshot(valid=False)
    # reorder positions into the expected joint order
    name_to_pos = dict(zip(names, positions, strict=True))
    ordered = [float(name_to_pos[n]) for n in expected]
    return JointSnapshot(positions=ordered, valid=True)


def construct_safety_command(
    *,
    plan: SafeStopPlan,
    last_action: Sequence[float] | None,
    joint_snapshot: JointSnapshot,
) -> list[list[float]]:
    """Build the per-channel safety command vectors.

    Returns one vector per channel, in plan order: zeros channels zeroed;
    hold channels prefer the last published target on matching positions,
    falling back to the latest joint observation ONLY where the channel
    position maps 1:1 to a joint index. Raises SafeStopError if a hold
    channel has neither -> the dispatcher stays FAILED/STOPPED and does not
    fabricate a value.
    """
    commands: list[list[float]] = []
    last = list(last_action) if last_action is not None else []
    for channel in plan.channels:
        n = len(channel.channel_names)
        if channel.safety_behavior == "zeros":
            commands.append([0.0] * n)
            continue
        # hold channel
        vec: list[float | None] = [None] * n
        # 1) prefer last published target (positional slice into last_action)
        if last:
            # last_action is a flat vector across all channels; find the offset
            offset = sum(len(c.channel_names) for c in plan.channels[: channel.spec_index])
            for i in range(n):
                idx = offset + i
                if idx < len(last) and _is_finite(last[idx]):
                    vec[i] = float(last[idx])
        # 2) fallback to joint observation for uniquely-mapped positions
        if joint_snapshot.valid:
            for i, jidx in enumerate(channel.joint_indices):
                if vec[i] is None and jidx is not None and jidx < len(joint_snapshot.positions):
                    val = joint_snapshot.positions[jidx]
                    if _is_finite(val):
                        vec[i] = float(val)
        # 3) any None remaining -> cannot fabricate -> fatal
        if any(v is None for v in vec):
            raise SafeStopError(
                f"hold channel {channel.spec_index} ({channel.topic}): "
                "no last action and no valid joint mapping for position(s); refusing to fabricate"
            )
        # apply clamp if declared
        if channel.clamp is not None:
            lo, hi = channel.clamp
            vec = [max(lo, min(hi, v)) for v in vec]  # type: ignore[misc]
        commands.append([float(v) for v in vec])  # type: ignore[misc]
    return commands


# ---------------------------------------------------------------------------
# helpers to read both SpecView and plain dicts
# ---------------------------------------------------------------------------


def _spec_names(spec: Any) -> list[str]:
    names = getattr(spec, "names", None)
    if names is None:
        selector = getattr(spec, "selector", None)
        if isinstance(selector, Mapping):
            names = selector.get("names")
    if names is None:
        selector = spec.get("selector") if isinstance(spec, Mapping) else None
        names = selector.get("names") if isinstance(selector, Mapping) else None
    if names is None:
        names = []
    return [str(n) for n in names]


def _spec_safety_behavior(spec: Any) -> str:
    sb = getattr(spec, "safety_behavior", None)
    if sb is None and isinstance(spec, Mapping):
        sb = spec.get("safety_behavior")
    return str(sb or "zeros")


def _spec_topic(spec: Any) -> str:
    return str(getattr(spec, "topic", None) or (spec.get("topic") if isinstance(spec, Mapping) else "") or "")


def _spec_ros_type(spec: Any) -> str:
    return str(getattr(spec, "ros_type", None) or (spec.get("ros_type") if isinstance(spec, Mapping) else "") or "")


def _spec_clamp(spec: Any) -> tuple[float, float] | None:
    clamp = getattr(spec, "clamp", None)
    if clamp is None and isinstance(spec, Mapping):
        clamp = spec.get("clamp")
    if clamp is None:
        return None
    if isinstance(clamp, list | tuple) and len(clamp) == 2:
        return (float(clamp[0]), float(clamp[1]))
    return None


def _contract_name_index_of(name: str, joint_order: Sequence[str]) -> int | None:
    """Resolve a contract action name against the canonical joint order.

    Named action selectors map directly by joint identity. Normalized policy
    tensor selectors use ``action.N``, where N indexes the same canonical joint
    order used when robot_config synthesizes the contract.
    """

    if name in joint_order:
        return joint_order.index(name)
    if not name.startswith("action."):
        return None
    suffix = name[len("action.") :]
    if not suffix.isdigit():
        return None
    idx = int(suffix)
    return idx if idx < len(joint_order) else None


def _is_finite(value: Any) -> bool:
    try:
        import math

        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


__all__ = [
    "JointSnapshot",
    "SafeStopChannel",
    "SafeStopError",
    "SafeStopPlan",
    "build_safe_stop_plan",
    "construct_safety_command",
    "validate_joint_state",
]
