"""Extension points of the mirror session. Every default here is a conservative no-op.

The session core calls these at fixed points of the WAIT / STANDBY / MIRRORING
skeleton and owns nothing they decide. Replace :class:`SessionHooks` (or
subclass it) and :class:`CoverDetector`; keep the signatures.

The defaults are the most conservative value available, not the mirror
behaviour: start and cancel never fire, the cover detector never fires, the
window pipeline passes nothing (the gate then holds the last command), wrap-up
finishes at once without moving, and cooldown never ends. Where a value had to
be sent anyway, the default says so below.

Inputs are plain dataclasses; ``mirror_session_node`` fills them from the
perception messages (``PearFrame``, ``ActorState``, the camera) once per tick.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from manipulation_execution.mirror_session.endpoints import HAND_JOINT_COUNT


@dataclass(frozen=True)
class UpperBodyCommand:
    """One tick of output: 14 arm radians, then 20 hand radians (left 10, right 10).

    Both in the joint order of the ``joint.arm_stream`` / ``joint.hand_stream``
    descriptions. The gate sends both channels on a tick or neither.
    ``source_stamp_s`` is the time of the perception frame the command was
    computed from, carried through the gate for latency accounting; None when
    the command does not come from a frame (holds, reset pose).
    """

    arm: tuple[float, ...]
    hand: tuple[float, ...]
    source_stamp_s: float | None = None


@dataclass(frozen=True)
class ActorSignal:
    """The latest ``ActorState``: who the perception is following.

    ``stamp_s`` is the camera frame's time; ``age_s`` is how long ago this
    process received it, so a stream that stopped publishing shows up here.
    """

    present: bool
    actor_id: int | None  # None: no actor (ActorState.actor_id == -1)
    stamp_s: float
    start_ready: bool = False
    reason: str = ""
    age_s: float = 0.0


@dataclass(frozen=True)
class PearWindow:
    """The ``PearFrame`` messages received since the previous tick, oldest first.

    Opaque to the core. Holding a longer window is up to the pipeline.
    """

    frames: tuple = ()


@dataclass(frozen=True)
class SessionInputs:
    """One tick of inputs. The node fills the perception fields and ``publish_enabled``;
    the core fills the rest before handing them to a hook.
    """

    window: PearWindow = PearWindow()
    actor: ActorSignal | None = None
    camera_frame: object | None = None
    measured_arm: tuple[float, ...] | None = None  # latest joint.state, arm_stream order
    measured_arm_age_s: float = math.inf  # how long ago it was received
    last_sent: UpperBodyCommand | None = None  # what the gate last sent, resends included
    publish_enabled: bool = False  # False: nothing reaches the joints, so they cannot move
    phase_elapsed_s: float = 0.0  # time spent in the current phase


@dataclass(frozen=True)
class HookContext:
    """What a business hook class is constructed with, next to its own config."""

    arm_joint_names: tuple[str, ...]  # 14, from the joint.arm_stream description
    hand_joint_names: tuple[str, ...]  # 20, from the joint.hand_stream description
    tick_hz: float
    wrap_up_budget_s: float
    publish_enabled: bool
    logger: Any


class WrapUpPurpose(str, Enum):
    APPROACH = "approach"  # WAIT -> mirror_start: from wherever the arm is to the reset pose
    CANCEL = "cancel"  # internal cancel: back to reset, then cooldown in STANDBY
    STOP = "stop"  # mirror_stop or cover: back to reset, then release


@dataclass(frozen=True)
class WrapUpContext:
    purpose: WrapUpPurpose
    last_command: UpperBodyCommand | None  # what the gate last sent while holding
    measured_arm: tuple[float, ...] | None  # latest joint.state, arm_stream order
    target: UpperBodyCommand | None  # reset_pose()
    elapsed_s: float
    budget_s: float  # the core finishes the wrap-up itself once this is spent


@dataclass(frozen=True)
class WrapUpStep:
    command: UpperBodyCommand | None  # None: hold the last command
    done: bool


class CoverDetector:
    """Camera covered (image present, very dark) == ``mirror_stop``. Default: never covered.

    Called only with a real frame: losing the camera stream is lost
    perception, not a cover.
    """

    def update(self, frame: object) -> bool:
        return False


class SessionHooks:
    """Every decision of the session. Each default is a conservative no-op."""

    cancel_reason: str = ""
    """Why the round is being cancelled. The core reads it right after
    ``cancel_criteria`` returned True (and the confirm time has passed) and
    reports it as the transition reason; empty is reported as ``criteria``."""

    def start_criteria(self, inputs: SessionInputs) -> bool:
        """STANDBY/ready: start mirroring? Must hold for ``start_confirm_s``. Default: never."""
        return False

    def cancel_criteria(self, inputs: SessionInputs, round_elapsed_s: float) -> bool:
        """MIRRORING: end this round (person gone, ID lost)? Must hold for ``cancel_confirm_s``.

        Set ``cancel_reason`` before returning True. Default: never.
        """
        return False

    def cooldown_over(self, elapsed_s: float) -> bool:
        """STANDBY after a cancel: may start criteria be evaluated again? Default: never."""
        return False

    def begin_round(self) -> None:
        """Entering MIRRORING: drop anything queued from an earlier round. Default: nothing queued."""

    def end_round(self) -> None:
        """Leaving the window pipeline (wrap-up begins, or the session drops to WAIT). Default: nothing to stop.

        Called exactly once per ``begin_round``. The session is already
        leaving the pipeline, so a raise here is recorded in the status, not
        acted on.
        """

    def retarget(self, inputs: SessionInputs) -> UpperBodyCommand | None:
        """Window pipeline, first stage. Default: passes nothing (the gate holds)."""
        return None

    def post_process(self, command: UpperBodyCommand) -> UpperBodyCommand | None:
        """Window pipeline, smoothing and limits. Default: passes nothing (the gate holds)."""
        return None

    def reset_pose(self, arm_at_takeover: Sequence[float] | None) -> UpperBodyCommand | None:
        """What STANDBY sends every tick.

        Default: hold the arm where it was measured at takeover, with all 20
        hand joints at zero (the sign convention of the left hand is not
        established, so no other hand target is sent). A fixed reset pose
        ignores the argument. None (no joint state yet) sends nothing.

        The default ``wrap_up`` finishes at once, so a pose that differs from
        where the arm is would be sent as a step: replace both together.
        """
        if arm_at_takeover is None:
            return None
        return UpperBodyCommand(tuple(float(v) for v in arm_at_takeover), (0.0,) * HAND_JOINT_COUNT)

    def wrap_up(self, context: WrapUpContext) -> WrapUpStep:
        """Interpolate toward ``context.target``. Default: done at once, no motion of its own."""
        return WrapUpStep(None, True)
