"""Whether a candidate is still about this visitor, and still worth saying now.

Three things, all about time and identity:

**Short-term association.** ``manipulation_execution.actor_selector`` picks the
highest-confidence person per frame and keeps no identity at all: its
``actor_id`` is always 0. That is enough to know somebody is there and not
enough to know it is the same somebody, so this module adds IoU, centre
movement and size continuity across frames. It is explicitly short-term: an
occlusion ends it, and nothing here survives one.

**Session boundaries.** A session is the de-duplication range, never an
identity. It ends three ways — an explicit hand-over to the next visitor, a
*confirmed* absence, or the maximum duration. A camera dropout is unknown
rather than absent, which is why ``ActorDecision.known`` matters: treating a
dropout as "they left" would reset coverage under a visitor who never moved.

**Validity.** A reply rests on a picture, and the picture ages. Age alone is
not the test: the target must still be there, the association must not have
broken, and the topic must not have been settled meanwhile. "Somebody is still
present" never makes an old event valid forever.

Times arrive as float seconds from a single clock the caller chooses, so this
module has no clock and no ROS types. The board demo's lesson is baked in: it
judges a dropped stream by *arrival* time, not by camera stamps, because those
depend on two machines agreeing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from embodied_common.visual_interaction_contract import (
    DEFAULT_ABSENT_RESET_SEC,
    DEFAULT_MAX_EVIDENCE_AGE_SEC,
    DEFAULT_MAX_SESSION_SEC,
    NO_ACTOR,
    SUPPRESS_AMBIGUOUS,
    SUPPRESS_DUPLICATE,
    SUPPRESS_STALE,
)

# How much two boxes must overlap to be the same person one frame later.
DEFAULT_MIN_IOU = 0.3
# How far the centre may move, as a fraction of the box's own width.
DEFAULT_MAX_CENTRE_SHIFT = 0.5
# How much the box may grow or shrink, as a ratio either way.
DEFAULT_MAX_SIZE_RATIO = 1.8
# How long a gap between frames may be before association is given up: beyond
# this the boxes are not comparable, whatever they look like.
DEFAULT_MAX_ASSOCIATION_GAP_SEC = 1.0
# How close a second person may be before the target is called ambiguous. As a
# fraction of the target box width, measured centre to centre.
DEFAULT_MIN_SEPARATION = 1.2

# Why a session ended, for diagnostics.
END_HANDOVER = "handover"
END_ABSENT = "absent"
END_MAX_DURATION = "max_duration"
SESSION_END_REASONS = (END_HANDOVER, END_ABSENT, END_MAX_DURATION)


@dataclass(frozen=True)
class Box:
    """A person box in source pixels, ``[x_min, y_min, x_max, y_max]``."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Box:
        if len(values) != 4:
            raise ValueError("a box needs four values")
        x_min, y_min, x_max, y_max = (float(value) for value in values)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("a box needs a positive width and height")
        return cls(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def iou(self, other: Box) -> float:
        overlap_x = min(self.x_max, other.x_max) - max(self.x_min, other.x_min)
        overlap_y = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)
        if overlap_x <= 0.0 or overlap_y <= 0.0:
            return 0.0
        intersection = overlap_x * overlap_y
        return intersection / (self.area + other.area - intersection)


@dataclass(frozen=True)
class AssociationResult:
    """Whether this frame continues the locked target.

    ``ambiguous`` is its own outcome rather than a kind of failure: it means
    the host cannot tell which person it is looking at, and the session must
    stay quiet instead of guessing. ``epoch_bumped`` says every window and
    in-flight result from before this frame is now void.
    """

    associated: bool
    ambiguous: bool
    epoch_bumped: bool
    reason: str = ""


@dataclass
class ActorTracker:
    """Short-term association on top of the identity-free actor selector.

    Not a tracker in the re-identification sense, and the naming is deliberate:
    it answers "is this the same box moving" for about a second, and gives up
    rather than guessing. A bumped ``epoch`` is the signal that everything
    collected under the previous one must be discarded.
    """

    min_iou: float = DEFAULT_MIN_IOU
    max_centre_shift: float = DEFAULT_MAX_CENTRE_SHIFT
    max_size_ratio: float = DEFAULT_MAX_SIZE_RATIO
    max_gap_sec: float = DEFAULT_MAX_ASSOCIATION_GAP_SEC
    min_separation: float = DEFAULT_MIN_SEPARATION
    epoch: int = 0
    _box: Box | None = None
    _stamp: float | None = None

    def __post_init__(self) -> None:
        if self.max_size_ratio <= 1.0:
            raise ValueError("max_size_ratio must be greater than 1")
        if self.max_gap_sec <= 0.0:
            raise ValueError("max_gap_sec must be positive")

    @property
    def locked(self) -> bool:
        return self._box is not None

    @property
    def box(self) -> Box | None:
        return self._box

    def update(
        self,
        *,
        present: bool,
        known: bool,
        box: Sequence[float] | None,
        stamp: float,
        others: Sequence[Sequence[float]] = (),
    ) -> AssociationResult:
        """Fold one frame in and say whether the target carried over.

        ``known`` comes straight from ``ActorDecision.known``: false means the
        detections are missing or stale, which is unknown, not absent. An
        unknown frame neither extends nor breaks the lock — it simply tells us
        nothing — so the gap check is what eventually ends it.
        """
        if not known:
            return AssociationResult(associated=False, ambiguous=False, epoch_bumped=False, reason="unknown")

        if not present or box is None:
            if self._box is None:
                return AssociationResult(associated=False, ambiguous=False, epoch_bumped=False, reason="absent")
            return self._drop("absent")

        current = Box.of(box)

        if others and self._is_ambiguous(current, others):
            # Two people close together: refuse rather than pick. A wrong pick
            # attributes one visitor's object to another.
            return AssociationResult(associated=False, ambiguous=True, epoch_bumped=False, reason=SUPPRESS_AMBIGUOUS)

        if self._box is None or self._stamp is None:
            self._box, self._stamp = current, stamp
            self.epoch += 1
            return AssociationResult(associated=True, ambiguous=False, epoch_bumped=True, reason="locked")

        gap = stamp - self._stamp
        if gap < 0.0 or gap > self.max_gap_sec:
            # Beyond the gap the boxes are not comparable however alike they
            # look, so a new lock starts instead of a continued one.
            return self._relock(current, stamp, "gap")

        if not self._continues(current):
            return self._relock(current, stamp, "moved")

        self._box, self._stamp = current, stamp
        return AssociationResult(associated=True, ambiguous=False, epoch_bumped=False, reason="continued")

    def _continues(self, current: Box) -> bool:
        previous = self._box
        assert previous is not None
        if previous.iou(current) < self.min_iou:
            return False
        shift = abs(current.centre[0] - previous.centre[0])
        if shift > self.max_centre_shift * previous.width:
            return False
        ratio = current.area / previous.area
        return 1.0 / self.max_size_ratio <= ratio <= self.max_size_ratio

    def _is_ambiguous(self, target: Box, others: Sequence[Sequence[float]]) -> bool:
        for other in others:
            candidate = Box.of(other)
            if candidate == target:
                continue
            separation = abs(candidate.centre[0] - target.centre[0])
            if separation < self.min_separation * target.width:
                return True
        return False

    def _relock(self, current: Box, stamp: float, reason: str) -> AssociationResult:
        self._box, self._stamp = current, stamp
        self.epoch += 1
        return AssociationResult(associated=True, ambiguous=False, epoch_bumped=True, reason=reason)

    def _drop(self, reason: str) -> AssociationResult:
        self._box = self._stamp = None
        self.epoch += 1
        return AssociationResult(associated=False, ambiguous=False, epoch_bumped=True, reason=reason)


@dataclass
class SessionWindow:
    """One interaction range: when it started and how it will end.

    Absence has to be *confirmed* to end a session. A queue of visitors rarely
    produces thirty quiet seconds, so the explicit hand-over is the normal path
    and the timer is the missed-hand-over guard. The maximum duration exists
    because without it a busy stand never resets at all, and every later
    visitor inherits the first one's covered topics.
    """

    session_id: str
    started_at: float
    absent_reset_sec: float = DEFAULT_ABSENT_RESET_SEC
    max_session_sec: float = DEFAULT_MAX_SESSION_SEC
    _absent_since: float | None = field(default=None)

    def observe(self, *, present: bool, known: bool, now: float) -> None:
        """Record presence. An unknown frame is not evidence of absence.

        A camera dropout or a stale detection leaves the absence timer exactly
        where it was: resetting coverage under a visitor who never moved is
        worse than waiting.
        """
        if not known:
            return
        if present:
            self._absent_since = None
        elif self._absent_since is None:
            self._absent_since = now

    def absent_for(self, now: float) -> float:
        if self._absent_since is None:
            return 0.0
        return max(0.0, now - self._absent_since)

    def expired_reason(self, now: float) -> str:
        """Why the session should end now, or an empty string to continue."""
        if self.absent_reset_sec > 0.0 and self.absent_for(now) >= self.absent_reset_sec:
            return END_ABSENT
        if self.max_session_sec > 0.0 and now - self.started_at >= self.max_session_sec:
            return END_MAX_DURATION
        return ""


@dataclass(frozen=True)
class Candidate:
    """A reply waiting to be spoken, and what it rests on."""

    event_id: str
    session_id: str
    actor_epoch: int
    evidence_stamp: float
    text: str


def candidate_validity(
    candidate: Candidate,
    *,
    session_id: str,
    actor_epoch: int,
    target_present: bool,
    topic_blocked: bool,
    now: float,
    max_age_sec: float = DEFAULT_MAX_EVIDENCE_AGE_SEC,
) -> str:
    """Return an empty string when the candidate may still be spoken.

    Otherwise a suppression reason. Checked again right before delivery, not
    only when the model finished: what matters is how old the *picture* is and
    whether it is still the same visitor.
    """
    if candidate.session_id != session_id or candidate.actor_epoch != actor_epoch:
        # A different session or epoch means the evidence belongs to somebody
        # else. Speaking it would attribute one visitor's object to another.
        return SUPPRESS_AMBIGUOUS
    if not target_present:
        return SUPPRESS_STALE
    if topic_blocked:
        return SUPPRESS_DUPLICATE
    if now - candidate.evidence_stamp > max_age_sec:
        return SUPPRESS_STALE
    return ""


__all__ = [
    "END_ABSENT",
    "END_HANDOVER",
    "END_MAX_DURATION",
    "NO_ACTOR",
    "SESSION_END_REASONS",
    "ActorTracker",
    "AssociationResult",
    "Box",
    "Candidate",
    "SessionWindow",
    "candidate_validity",
]
