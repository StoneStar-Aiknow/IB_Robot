"""Evidence-gated online movement and watched-target reacquisition."""

from dataclasses import dataclass

import numpy as np

from .association import LifecycleEvidence, SemanticTracker, cosine_similarity


@dataclass
class _MoveCandidate:
    position: np.ndarray
    confirmations: int
    stamp_ns: int | None = None
    session_id: str = ""


class OnlineLifecycleCoordinator:
    def __init__(
        self,
        tracker: SemanticTracker,
        *,
        move_distance_m: float = 0.45,
        move_stability_m: float = 0.1,
        move_confirmations: int = 2,
        move_confirmation_max_gap_sec: float = 1.0,
    ):
        if (
            move_distance_m <= 0.0
            or move_stability_m < 0.0
            or move_confirmations <= 0
            or move_confirmation_max_gap_sec <= 0.0
        ):
            raise ValueError("movement thresholds and confirmation count must be positive")
        self.tracker = tracker
        self.move_distance_m = move_distance_m
        self.move_stability_m = move_stability_m
        self.move_confirmations = move_confirmations
        self.move_confirmation_max_gap_ns = int(move_confirmation_max_gap_sec * 1e9)
        self._move_candidates: dict[str, _MoveCandidate] = {}
        self._search_attempts: dict[str, int] = {}

    def observe_remote_identity(
        self,
        object_id: str,
        position: np.ndarray,
        embedding: np.ndarray,
    ) -> bool:
        track = self.tracker.tracks[object_id]
        similarity = cosine_similarity(track.embedding, embedding)
        if similarity is None or similarity < self.tracker.embedding_similarity_threshold:
            return False
        return self._observe_confirmed_identity(object_id, position, similarity=similarity, source="embedding")

    def observe_tracked_identity(
        self,
        object_id: str,
        position: np.ndarray,
        *,
        stamp_ns: int,
        session_id: str,
    ) -> bool:
        """Confirm movement when a tracker carries the persistent semantic object ID."""
        if object_id not in self.tracker.tracks:
            return False
        return self._observe_confirmed_identity(
            object_id,
            position,
            similarity=None,
            source="track_state",
            stamp_ns=stamp_ns,
            session_id=session_id,
        )

    def reset_move_candidate(self, object_id: str) -> None:
        self._move_candidates.pop(object_id, None)

    def _observe_confirmed_identity(
        self,
        object_id: str,
        position: np.ndarray,
        *,
        similarity: float | None,
        source: str,
        stamp_ns: int | None = None,
        session_id: str = "",
    ) -> bool:
        track = self.tracker.tracks[object_id]
        position = np.asarray(position, dtype=np.float64)
        if np.linalg.norm(position - track.position) <= self.move_distance_m:
            self._move_candidates.pop(object_id, None)
            return False
        candidate = self._move_candidates.get(object_id)
        invalid_sequence = (
            candidate is not None
            and stamp_ns is not None
            and candidate.stamp_ns is not None
            and (stamp_ns <= candidate.stamp_ns or stamp_ns - candidate.stamp_ns > self.move_confirmation_max_gap_ns)
        )
        session_changed = candidate is not None and candidate.session_id != session_id
        if (
            candidate is None
            or invalid_sequence
            or session_changed
            or np.linalg.norm(candidate.position - position) > self.move_stability_m
        ):
            self._move_candidates[object_id] = _MoveCandidate(position.copy(), 1, stamp_ns, session_id)
            return False
        candidate.position = (candidate.position * candidate.confirmations + position) / (candidate.confirmations + 1)
        candidate.confirmations += 1
        candidate.stamp_ns = stamp_ns
        if candidate.confirmations < self.move_confirmations:
            return False
        self.tracker.mark_moved(
            object_id,
            candidate.position,
            LifecycleEvidence(
                identity_confirmed=True,
                geometry_confirmed=True,
                details={
                    "confirmations": candidate.confirmations,
                    "identity_source": source,
                    **({} if not session_id else {"tracking_session_id": session_id}),
                    **({} if similarity is None else {"embedding_similarity": similarity}),
                },
            ),
        )
        self._move_candidates.pop(object_id, None)
        return True

    def mark_expected_region_empty(self, object_id: str, details: dict | None = None) -> None:
        self.tracker.mark_missing(
            object_id,
            LifecycleEvidence(geometry_confirmed=True, details=details or {"expected_region_empty": True}),
        )

    def begin_watch(self, object_id: str) -> None:
        if object_id not in self.tracker.tracks:
            raise KeyError(object_id)
        self._search_attempts[object_id] = 0

    def record_search_failure(self, object_id: str, *, max_attempts: int, details: dict | None = None) -> bool:
        if max_attempts <= 0:
            raise ValueError("maximum search attempts must be positive")
        attempts = self._search_attempts.get(object_id, 0) + 1
        self._search_attempts[object_id] = attempts
        if attempts < max_attempts:
            return False
        self.tracker.mark_lost(
            object_id,
            LifecycleEvidence(
                search_exhausted=True,
                details={"attempts": attempts, **(details or {})},
            ),
        )
        return True

    def record_reacquired(self, object_id: str) -> None:
        self.tracker.transition(
            object_id,
            "observed",
            LifecycleEvidence(identity_confirmed=True, geometry_confirmed=True, details={"reacquired": True}),
        )
        self._search_attempts.pop(object_id, None)
