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

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from manipulation_execution.mirror_session.endpoints import HAND_JOINT_COUNT


@dataclass(frozen=True)
class UpperBodyCommand:
    """One tick of output: 14 arm radians, then 20 hand radians (left 10, right 10).

    Both in the joint order of the ``joint.arm_stream`` / ``joint.hand_stream``
    descriptions. The gate sends both channels on a tick or neither.
    """

    arm: tuple[float, ...]
    hand: tuple[float, ...]


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
    window: PearWindow = PearWindow()
    actor: ActorSignal | None = None
    camera_frame: object | None = None


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

    def start_criteria(self, inputs: SessionInputs) -> bool:
        """STANDBY, outside cooldown: start mirroring without a request? Default: never."""
        return False

    def cancel_criteria(self, inputs: SessionInputs, round_elapsed_s: float) -> bool:
        """MIRRORING: end this round (max_round_s, person gone, ID lost)? Default: never."""
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
