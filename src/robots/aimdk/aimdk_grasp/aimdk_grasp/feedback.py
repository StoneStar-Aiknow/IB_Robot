"""Robot joint feedback associated with an image time.

A grasp observation records the arm, head and finger pose at the moment the
image was taken: the head pose places the camera, the locked right arm and head
enter the collision model, and the finger pose decides whether the hand is open.

The runtime publishes these as three JointState views (``joint.state`` for the
arms, ``joint.head_state`` for the head, ``joint.hand_state`` for the
fingers), each stamped with its own oldest measurement. This module keeps a short
history of each view and assembles, for an image stamp, the samples nearest to
it. It is ROS-free so it can be tested and replayed without a graph; the
subscription side is :mod:`aimdk_grasp.feedback_ros`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .joints import BODY_JOINTS, LEFT_HAND, RIGHT_HAND, check_side
from .joints import installed_hands as parse_installed_hands

#: The runtime interface IDs the feedback is assembled from, and the joints each
#: must provide. Grasping needs all three; an explicitly handless consumer
#: requires only the body and head views. Missing finger feedback never proves
#: that a hand is open or physically absent.
VIEWS = ("joint.state", "joint.head_state", "joint.hand_state")

#: A sample further than this from the image stamp does not describe the image.
MAX_IMAGE_OFFSET_NS = 50_000_000
#: The views of one assembled feedback may not straddle more than this.
MAX_VIEW_SPREAD_NS = 100_000_000
#: The runtime publishes its status on a one-second heartbeat (aimdk_robot
#: runtime_node) and on every state change. A verdict older than this when the
#: feedback is assembled no longer proves the runtime is alive: the views come
#: from the data plane's own process, so they keep updating after the runtime's
#: process is gone, and joints alone must not read as a healthy runtime.
MAX_STATUS_AGE_S = 5.0
_HISTORY = 120


@dataclass(frozen=True)
class Feedback:
    """Joint positions at one time.

    ``received_at`` is the oldest view stamp in seconds: the assembled pose is
    never newer than its stalest part. ``healthy`` is the runtime's own verdict
    (ACTIVE, no faults) and that the verdict is still fresh: a status the
    middleware did not time, or one older than :data:`MAX_STATUS_AGE_S`, is
    reported unhealthy, because the views keep flowing from the data plane
    after the runtime's own process has stopped; joint-level faults are
    reported by the runtime, not re-derived here.

    Each hand is None when the hand view does not carry it (a hand that is not
    installed, ``backend_options.installed_hands``), and :meth:`hand` refuses
    it. Whoever needs a hand asks for it by side, so a missing hand fails there
    instead of being skipped.
    """

    received_at: float
    positions: Mapping[str, float]
    left_hand: np.ndarray | None
    healthy: bool = True
    right_hand: np.ndarray | None = None

    def hand(self, side: str) -> np.ndarray:
        """Native finger positions of ``side``'s hand, in ``hand_joints(side)`` order."""
        hand = self.left_hand if check_side(side) == "left" else self.right_hand
        if hand is None:
            raise ValueError(f"Feedback lacks {side} hand joints")
        return hand

    def vector(self, names: Iterable[str]) -> np.ndarray:
        result = np.array([self.positions[name] for name in names], dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite feedback")
        return result


class FeedbackHistory:
    """Bounded per-view history; thread-safe, so subscriptions may run in parallel.

    ``required_hands`` is the explicit mounting configuration. An empty tuple
    consumes no hand view; ``None`` preserves the legacy reader that requires a
    hand view and accepts whichever complete hands it carries. Grasp and restore
    consumers always pass their mounting configuration.
    """

    def __init__(
        self,
        *,
        history: int = _HISTORY,
        max_status_age_s: float = MAX_STATUS_AGE_S,
        required_hands: tuple[str, ...] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self.depth = int(history)
        self.required_hands = None if required_hands is None else parse_installed_hands(required_hands)
        self.views = VIEWS[:2] if self.required_hands == () else VIEWS
        self._views: dict[str, deque[tuple[int, dict[str, float]]]] = {
            view: deque(maxlen=history) for view in self.views
        }
        self._healthy = False
        self._status_at: float | None = None
        self._max_status_age_s = float(max_status_age_s)

    def record(self, view: str, stamp_ns: int, names: Iterable[str], positions: Iterable[float]) -> bool:
        """Add one sample; returns False for an unknown view, an unset stamp or an out-of-order sample."""
        if view not in self._views or stamp_ns <= 0:
            return False
        sample = dict(zip((str(name) for name in names), (float(value) for value in positions), strict=True))
        with self._lock:
            history = self._views[view]
            if history and stamp_ns <= history[-1][0]:
                return False
            history.append((int(stamp_ns), sample))
        return True

    def set_healthy(self, healthy: bool, received_at: float | None) -> None:
        """Record the runtime's verdict and when its message was received.

        ``received_at`` is the receipt time the on-demand middleware recorded
        for the status message, in monotonic seconds; ``None`` is a message it
        did not time, which can never count as fresh.
        """
        with self._lock:
            self._healthy = bool(healthy)
            self._status_at = None if received_at is None else float(received_at)

    def _healthy_now(self) -> bool:
        # Called under the lock; a verdict is healthy only while it is provably fresh.
        if not self._healthy or self._status_at is None:
            return False
        return time.monotonic() - self._status_at <= self._max_status_age_s

    def latest(self) -> Feedback:
        """The newest sample of every view."""
        with self._lock:
            samples = {view: history[-1] for view, history in self._views.items() if history}
            healthy = self._healthy_now()
        return _assemble(samples, healthy, self.required_hands)

    def at(self, image_stamp_ns: int) -> Feedback:
        """The sample of every view nearest to ``image_stamp_ns``, each within 50 ms of it."""
        with self._lock:
            samples = {
                view: min(history, key=lambda sample: abs(sample[0] - image_stamp_ns))
                for view, history in self._views.items()
                if history
            }
            healthy = self._healthy_now()
        late = {view: stamp - image_stamp_ns for view, (stamp, _) in samples.items()}
        if any(abs(offset) > MAX_IMAGE_OFFSET_NS for offset in late.values()):
            offsets = {view: round(offset * 1e-6, 1) for view, offset in late.items()}
            raise ValueError(f"Image has no joint feedback within 50 ms (nearest, ms: {offsets})")
        return _assemble(samples, healthy, self.required_hands)


def _assemble(
    samples: Mapping[str, tuple[int, dict[str, float]]], healthy: bool, required_hands: tuple[str, ...] | None = None
) -> Feedback:
    views = VIEWS[:2] if required_hands == () else VIEWS
    missing_views = sorted(set(views) - set(samples))
    if missing_views:
        raise ValueError(f"No feedback yet from {missing_views}")
    stamps = [stamp for stamp, _ in samples.values()]
    if max(stamps) - min(stamps) > MAX_VIEW_SPREAD_NS:
        raise ValueError("Feedback views are not synchronized")
    positions: dict[str, float] = {}
    for view in ("joint.state", "joint.head_state"):
        for name, value in samples[view][1].items():
            if name in positions:
                raise ValueError(f"Duplicate joint feedback for {name}")
            positions[name] = value
    missing = [name for name in BODY_JOINTS if name not in positions]
    if missing:
        raise ValueError(f"Feedback lacks body joints {missing}")
    hand = samples.get("joint.hand_state", (0, {}))[1]
    hands = {}
    for side, names in (("left", LEFT_HAND), ("right", RIGHT_HAND)):
        present = [name for name in names if name in hand]
        if not present:
            hands[side] = None
            continue
        if len(present) != len(names):
            raise ValueError(f"Feedback lacks {side} hand joints {[name for name in names if name not in hand]}")
        values = np.array([hand[name] for name in names], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {side} hand feedback")
        hands[side] = values
    if required_hands is not None:
        for side in required_hands:
            if hands[side] is None:
                raise ValueError(f"Feedback lacks installed {side} hand joints")
        unexpected = [side for side, values in hands.items() if values is not None and side not in required_hands]
        if unexpected:
            raise ValueError(f"Feedback carries hands declared not installed: {unexpected}")
    elif hands["left"] is None and hands["right"] is None:
        raise ValueError("Feedback carries no hand joints")
    feedback = Feedback(min(stamps) * 1e-9, positions, hands["left"], healthy, hands["right"])
    feedback.vector(BODY_JOINTS)
    return feedback
