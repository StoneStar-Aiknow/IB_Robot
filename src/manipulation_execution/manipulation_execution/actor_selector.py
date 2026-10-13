"""Choose which person in the frame is the actor the mirror follows.

The stream node asks this once per frame it runs PEAR on, handing over the
newest YOLOX result. The answer decides the crop PEAR sees and is published as
``ibrobot_msgs/ActorState`` so a session stage can decide when to start.

This implementation keeps no identity across frames: the actor is the
highest-confidence qualifying person, ``actor_id`` is always 0 when there is
one, and ``start_ready`` repeats ``actor_present``. A selector that tracks one
person through crossings, or requires a start posture or a standing zone,
replaces ``update`` and keeps ``ActorDecision`` as its output; ``update`` is a
method of a class so that such a selector can keep state.

Staleness is part of the contract, not of this implementation. YOLOX refreshes far less
often than frames arrive and its refresh can fail, so a decision may rest on
detections from an older frame. Once those are more than ``stale_sec`` older
than the frame being decided, the actor is *unknown*, and unknown never counts
as present. Both stamps are the camera's observation times, so the comparison
does not depend on how the camera clock relates to this node's clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image

from ibrobot_msgs.msg import DetectionArray
from manipulation_execution.hri_vision import qualifying_persons, select_person_detection

NO_ACTOR = -1

# ``reason`` values. The first two are known outcomes; the last two are unknown.
REASON_ACTOR = "highest_confidence_person"
REASON_NO_PERSON = "no_person"
REASON_NO_DETECTIONS = "no_detections"
REASON_STALE = "stale_detections"
UNKNOWN_REASONS = frozenset({REASON_NO_DETECTIONS, REASON_STALE})


@dataclass(frozen=True)
class ActorDecision:
    """The actor for one frame. Mirrors ``ibrobot_msgs/ActorState``."""

    # Observation time of the frame this decision is for.
    stamp: Time
    # Observation time of the detections it rests on; None before the first.
    detections_stamp: Time | None
    persons_count: int
    actor_present: bool
    # Equal to actor_present in this implementation (no start rule).
    start_ready: bool
    actor_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    reason: str

    @property
    def known(self) -> bool:
        return self.reason not in UNKNOWN_REASONS


def stamp_seconds(stamp: Time) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class ActorSelector:
    """Pass-through actor selection: highest-confidence person, id 0."""

    def __init__(self, person_confidence_threshold: float, stale_sec: float) -> None:
        stale = float(stale_sec)
        if not math.isfinite(stale) or stale <= 0.0:
            raise ValueError("actor_stale_sec must be finite and greater than zero")
        self._threshold = float(person_confidence_threshold)
        self._stale_sec = stale

    def update(self, frame: Image, detections: DetectionArray | None) -> ActorDecision:
        """Decide the actor for ``frame`` from the newest YOLOX ``detections``."""
        if detections is None:
            return self._without_actor(frame, None, REASON_NO_DETECTIONS)
        age = stamp_seconds(frame.header.stamp) - stamp_seconds(detections.header.stamp)
        if age > self._stale_sec:
            return self._without_actor(frame, detections.header.stamp, REASON_STALE)
        persons = qualifying_persons(list(detections.detections), self._threshold)
        actor = select_person_detection(persons, self._threshold)
        if actor is None:
            return self._without_actor(frame, detections.header.stamp, REASON_NO_PERSON)
        return ActorDecision(
            stamp=frame.header.stamp,
            detections_stamp=detections.header.stamp,
            persons_count=len(persons),
            actor_present=True,
            start_ready=True,
            actor_id=0,
            bbox=tuple(float(value) for value in actor.bbox),
            confidence=float(actor.confidence),
            reason=REASON_ACTOR,
        )

    @staticmethod
    def _without_actor(frame: Image, detections_stamp: Time | None, reason: str) -> ActorDecision:
        return ActorDecision(
            stamp=frame.header.stamp,
            detections_stamp=detections_stamp,
            persons_count=0,
            actor_present=False,
            start_ready=False,
            actor_id=NO_ACTOR,
            bbox=(0.0, 0.0, 0.0, 0.0),
            confidence=0.0,
            reason=reason,
        )


__all__ = [
    "NO_ACTOR",
    "REASON_ACTOR",
    "REASON_NO_DETECTIONS",
    "REASON_NO_PERSON",
    "REASON_STALE",
    "UNKNOWN_REASONS",
    "ActorDecision",
    "ActorSelector",
    "stamp_seconds",
]
