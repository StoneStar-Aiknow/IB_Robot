"""Episode context: the request each evaluation decision sends to the Agent.

The runner asks the Agent for one skill per decision. This module turns a whitelisted robot
observation into that request: the task instruction, the robot state, the tick budget and the
recipe's rules. Nothing is carried from one decision to the next, and the runner gives every
decision a fresh Agent session, so neither the request nor the planner's conversation holds an
earlier decision. Memory is the Agent's to add, measured against this.

It has no ROS imports and never sees evaluator data: its only inputs are
:class:`EpisodeObservation` (built through an allowlist), task knowledge (:class:`RecipeFacts`)
and body facts (:class:`BodyFacts`, built from the robot configuration).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "CANCELLED_BEFORE_EXECUTION", "ANSWERED", "UNKNOWN"})

#: The only observation fields an episode context may receive.
OBSERVATION_FIELDS = ("instruction", "tick", "step_lim", "arms")
#: The only per-arm fields: tool centre point position (m) and quaternion (x, y, z, w) in the
#: runtime's pose frame, normalized gripper opening and wrist joint angle (rad).
ARM_OBSERVATION_FIELDS = ("tcp_xyz", "tcp_quat", "gripper_opening", "wrist_joint_rad")

DEFAULT_APPROACH_MATCH_DEG = 8.0


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_vector(value: Any, size: int) -> tuple[float, ...] | None:
    if value is None or isinstance(value, str | bytes) or not hasattr(value, "__len__") or len(value) != size:
        return None
    numbers = tuple(_optional_float(v) for v in value)
    return None if any(v is None for v in numbers) else numbers


@dataclass(frozen=True)
class ArmObservation:
    """Proprioception of one arm; every field may be unknown (``None``)."""

    tcp_xyz: tuple[float, float, float] | None = None
    tcp_quat: tuple[float, float, float, float] | None = None
    gripper_opening: float | None = None
    wrist_joint_rad: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> ArmObservation:
        """Build from a mapping, keeping only :data:`ARM_OBSERVATION_FIELDS`."""
        data = data if isinstance(data, Mapping) else {}
        return cls(
            tcp_xyz=_optional_vector(data.get("tcp_xyz"), 3),
            tcp_quat=_optional_vector(data.get("tcp_quat"), 4),
            gripper_opening=_optional_float(data.get("gripper_opening")),
            wrist_joint_rad=_optional_float(data.get("wrist_joint_rad")),
        )


@dataclass(frozen=True)
class EpisodeObservation:
    """The whitelisted observation one request is built from (no score, reward or object truth)."""

    instruction: str
    tick: int | None = None
    step_lim: int | float | None = None
    arms: Mapping[str, ArmObservation] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EpisodeObservation:
        """Build from a mapping, dropping every key outside :data:`OBSERVATION_FIELDS`."""
        tick = data.get("tick")
        step_lim = data.get("step_lim")
        arms = data.get("arms") if isinstance(data.get("arms"), Mapping) else {}
        return cls(
            instruction=str(data.get("instruction") or ""),
            tick=int(tick) if isinstance(tick, int | float) and not isinstance(tick, bool) else None,
            step_lim=step_lim if isinstance(step_lim, int | float) and not isinstance(step_lim, bool) else None,
            arms={str(side): ArmObservation.from_mapping(arm) for side, arm in arms.items()},
        )


@dataclass(frozen=True)
class BodyFacts:
    """Robot body facts the context needs, built from the robot configuration."""

    sides: tuple[str, ...]
    approach_presets: Mapping[str, tuple[float, float, float, float]] = field(default_factory=dict)
    approach_axis: str = "z"
    approach_match_deg: float = DEFAULT_APPROACH_MATCH_DEG

    def __post_init__(self) -> None:
        if not self.sides:
            raise ValueError("body needs at least one arm side")
        if self.approach_axis not in ("x", "y", "z"):
            raise ValueError(f"approach axis must be x, y or z, got {self.approach_axis!r}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BodyFacts:
        presets = data.get("approach_presets") or {}
        return cls(
            sides=tuple(str(s) for s in data.get("sides") or ()),
            approach_presets={str(k): tuple(float(v) for v in q) for k, q in presets.items()},
            approach_axis=str(data.get("approach_axis") or "z"),
            approach_match_deg=float(data.get("approach_match_deg", DEFAULT_APPROACH_MATCH_DEG)),
        )


@dataclass(frozen=True)
class RecipeFacts:
    """Task knowledge the context consumes: the rules text and the fingertip height reference."""

    guide: str
    fingers_on_table_z: float
    done_word: str = "DONE"
    max_idle_replies: int = 3

    def __post_init__(self) -> None:
        if not self.guide.strip():
            raise ValueError("recipe guide is empty")
        if not self.done_word or self.done_word != self.done_word.upper():
            raise ValueError("recipe done word must be non-empty upper case")
        if self.max_idle_replies < 1:
            raise ValueError("max_idle_replies must be at least 1")


@dataclass(frozen=True)
class EpisodeInfo:
    """Public identity of the episode a context is reset to."""

    episode_id: str
    generation: int | None = None
    tag: str | None = None  # perturbs the prompt so repeated episodes are independent samples


# ---------------------------------------------------------------------------- pure helpers

_HALF_TURN = {"x": (1.0, 0.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0, 0.0), "z": (0.0, 0.0, 1.0, 0.0)}


def _quat_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _axis_vector(quat: tuple[float, ...], axis: str) -> tuple[float, float, float]:
    x, y, z, w = quat
    if axis == "x":
        return (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    if axis == "y":
        return (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def approach_name(
    quat: tuple[float, float, float, float] | None,
    presets: Mapping[str, tuple],
    axis: str = "z",
    match_deg: float = DEFAULT_APPROACH_MATCH_DEG,
) -> str:
    """The declared approach the gripper is in, or 'custom (points dx,dy,dz)' (proprioception only)."""
    if quat is None:
        return "?"
    best, best_deg = "", 180.0
    for name, preset in presets.items():
        # A parallel gripper turned 180 deg about its approach axis grasps the same way.
        for candidate in (tuple(preset), _quat_mul(tuple(preset), _HALF_TURN[axis])):
            dot = min(1.0, abs(sum(a * b for a, b in zip(quat, candidate, strict=True))))
            deg = math.degrees(2.0 * math.acos(dot))
            if deg < best_deg:
                best, best_deg = name, deg
    if best and best_deg <= match_deg:
        return best
    return "custom (points {:+.2f},{:+.2f},{:+.2f})".format(*(round(v, 2) + 0.0 for v in _axis_vector(quat, axis)))


def state_view(observation: EpisodeObservation, body: BodyFacts) -> dict[str, Any]:
    """Per-arm grasp point (mm-rounded), gripper opening, named pose and wrist angle (deg)."""
    arms = {side: observation.arms.get(side) or ArmObservation() for side in body.sides}
    return {
        "ee": {side: None if arm.tcp_xyz is None else [round(v, 3) for v in arm.tcp_xyz] for side, arm in arms.items()},
        "grippers": {side: arm.gripper_opening for side, arm in arms.items()},
        "pose": {
            side: approach_name(arm.tcp_quat, body.approach_presets, body.approach_axis, body.approach_match_deg)
            for side, arm in arms.items()
        },
        "wrist_deg": {
            side: None if arm.wrist_joint_rad is None else round(math.degrees(arm.wrist_joint_rad), 1)
            for side, arm in arms.items()
        },
    }


def state_text(observation: EpisodeObservation, body: BodyFacts, recipe: RecipeFacts) -> tuple[str, dict[str, Any]]:
    """The ROBOT STATE lines and the state view they were written from."""
    state = state_view(observation, body)
    lines = []
    for side in body.sides:
        xyz = state["ee"][side]
        opening = state["grippers"][side]
        where = "unknown" if xyz is None else f"x={xyz[0]:.3f} y={xyz[1]:.3f} z={xyz[2]:.3f}"
        fingers = "?" if xyz is None else f"{xyz[2] - recipe.fingers_on_table_z:.3f}"
        wrist = state["wrist_deg"][side]
        lines.append(
            f"{side} gripper: grasp point {where} (fingers_above_table={fingers} m), opening="
            f"{'?' if opening is None else round(opening, 2)} (1=open, 0=closed), pose={state['pose'][side]}, "
            f"wrist joint={'?' if wrist is None else f'{wrist:+.0f}'} deg"
        )
    return "\n".join(lines), state


def budget_text(observation: EpisodeObservation) -> str:
    return (
        f"TIME: {observation.tick} of {observation.step_lim} control ticks used (a move costs about 8 ticks, "
        "a close or open 15); when the budget runs out the episode fails."
    )


# ---------------------------------------------------------------------------- the context


class EpisodeContext:
    """Builds each decision's request from the current observation alone."""

    def __init__(self, recipe: RecipeFacts, body: BodyFacts):
        self.recipe = recipe
        self.body = body
        self._episode = EpisodeInfo(episode_id="")

    @property
    def max_idle_replies(self) -> int:
        return self.recipe.max_idle_replies

    def reset(self, episode: EpisodeInfo) -> None:
        self._episode = episode

    def state_view(self, observation: EpisodeObservation) -> dict[str, Any]:
        return state_view(observation, self.body)

    def is_done_request(self, result: Mapping[str, Any]) -> bool:
        """An ANSWERED reply naming the done word: an end request, never a success signal."""
        return (
            result.get("state") == "ANSWERED" and self.recipe.done_word in str(result.get("user_message", "")).upper()
        )

    def build_request_text(self, observation: EpisodeObservation) -> str:
        text, _ = state_text(observation, self.body, self.recipe)
        head = f"TASK: {observation.instruction}\nROBOT STATE:\n{text}\n{budget_text(observation)}\n"
        if self._episode.tag:
            head = f"EPISODE {self._episode.tag}\n{head}"
        return f"{head}RULES: {self.recipe.guide}"
