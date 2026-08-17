"""Persistent semantic-object association and fusion."""

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

FROZEN = "frozen"
OBSERVED = "observed"
STALE = "stale"
MISSING = "missing"
MOVED = "moved"
LOST = "lost"
ACTION_READY_STATES = {OBSERVED, MOVED}
LIFECYCLE_STATES = {FROZEN, OBSERVED, STALE, MISSING, MOVED, LOST}
_ALLOWED_TRANSITIONS = {
    FROZEN: {OBSERVED, MOVED},
    OBSERVED: {OBSERVED, MOVED, STALE, MISSING},
    STALE: {OBSERVED, MOVED, LOST},
    MISSING: {OBSERVED, MOVED, LOST},
    MOVED: {MOVED, OBSERVED, STALE},
    LOST: {OBSERVED, MOVED},
}


@dataclass(frozen=True)
class LifecycleEvidence:
    identity_confirmed: bool = False
    geometry_confirmed: bool = False
    search_exhausted: bool = False
    freshness_expired: bool = False
    details: dict = field(default_factory=dict)


@dataclass
class SemanticObservation:
    label: str
    confidence: float
    position: np.ndarray
    size: np.ndarray
    point_count: int
    stamp_ns: int
    embedding: np.ndarray | None = None
    attributes: dict = field(default_factory=dict)
    canonical_label: str = ""
    map_version: str = ""
    session_id: str = ""
    source_frame: str = ""
    model_versions: dict = field(default_factory=dict)
    semantic_identities: dict = field(default_factory=dict)
    deployment_provenance: dict = field(default_factory=dict)
    mapping_run_id: str = ""
    label_candidates: tuple[tuple[str, float], ...] = ()


@dataclass
class SemanticTrack:
    object_id: str
    label: str
    confidence: float
    position: np.ndarray
    size: np.ndarray
    point_count: int
    first_seen_ns: int
    last_seen_ns: int
    observation_count: int = 1
    embedding: np.ndarray | None = None
    attributes: dict = field(default_factory=dict)
    canonical_label: str = ""
    state: str = OBSERVED
    map_version: str = ""
    session_id: str = ""
    object_version: int = 1
    model_versions: dict = field(default_factory=dict)
    lifecycle_evidence: dict = field(default_factory=dict)
    semantic_identities: dict = field(default_factory=dict)
    deployment_provenance: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.state in ACTION_READY_STATES

    @active.setter
    def active(self, value: bool) -> None:
        self.state = OBSERVED if value else STALE


def normalize_embedding(embedding: np.ndarray | None) -> np.ndarray | None:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else None


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    left, right = normalize_embedding(left), normalize_embedding(right)
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


class SemanticTracker:
    def __init__(
        self,
        association_distance_m: float = 0.45,
        embedding_similarity_threshold: float = 0.72,
        position_weight: float = 0.55,
        max_size_ratio: float = 4.0,
        label_switch_confidence_margin: float = 0.05,
        label_recurrence_count_ratio: float = 3.0,
        label_high_confidence_override_margin: float = 0.08,
        stale_after_sec: float = 10.0,
    ):
        self.association_distance_m = association_distance_m
        self.embedding_similarity_threshold = embedding_similarity_threshold
        self.position_weight = position_weight
        self.max_size_ratio = max_size_ratio
        self.label_switch_confidence_margin = label_switch_confidence_margin
        self.label_recurrence_count_ratio = label_recurrence_count_ratio
        self.label_high_confidence_override_margin = label_high_confidence_override_margin
        self.stale_after_ns = int(stale_after_sec * 1e9)
        self.tracks: dict[str, SemanticTrack] = {}

    def add_track(self, track: SemanticTrack) -> None:
        if track.state not in LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle state: {track.state}")
        track.embedding = normalize_embedding(track.embedding)
        self.tracks[track.object_id] = track

    def update(self, observation: SemanticObservation, excluded_object_ids: set[str] | None = None) -> SemanticTrack:
        observation.embedding = normalize_embedding(observation.embedding)
        match = self._best_match(observation, excluded_object_ids or set())
        if match is None:
            track = SemanticTrack(
                object_id=str(uuid.uuid4()),
                label=observation.label,
                confidence=observation.confidence,
                position=np.asarray(observation.position, dtype=np.float64),
                size=np.asarray(observation.size, dtype=np.float64),
                point_count=observation.point_count,
                first_seen_ns=observation.stamp_ns,
                last_seen_ns=observation.stamp_ns,
                embedding=observation.embedding,
                attributes={
                    **observation.attributes,
                    "label_evidence": {observation.label.casefold(): 1},
                    "label_score_evidence": {observation.label.casefold(): float(observation.confidence)},
                    "label_max_confidence": {observation.label.casefold(): float(observation.confidence)},
                    "label_candidate_evidence": self._candidate_evidence(observation.label_candidates),
                },
                canonical_label=observation.canonical_label or observation.label.casefold(),
                map_version=observation.map_version,
                session_id=observation.session_id,
                model_versions=dict(observation.model_versions),
                semantic_identities=dict(observation.semantic_identities),
                deployment_provenance=dict(observation.deployment_provenance),
            )
            self.tracks[track.object_id] = track
            return track

        alpha = 1.0 / min(match.observation_count + 1, 5)
        match.position = (1.0 - alpha) * match.position + alpha * observation.position
        match.size = (1.0 - alpha) * match.size + alpha * observation.size
        match.point_count = observation.point_count
        match.last_seen_ns = observation.stamp_ns
        match.observation_count += 1
        self.transition(
            match.object_id,
            OBSERVED,
            LifecycleEvidence(identity_confirmed=True, geometry_confirmed=True, details={"source": "association"}),
        )
        label_evidence = dict(match.attributes.get("label_evidence", {}))
        label_score_evidence = dict(match.attributes.get("label_score_evidence", {}))
        label_max_confidence = dict(match.attributes.get("label_max_confidence", {}))
        normalized_observation_label = observation.label.casefold()
        label_evidence[normalized_observation_label] = int(label_evidence.get(normalized_observation_label, 0)) + 1
        label_score_evidence[normalized_observation_label] = float(
            label_score_evidence.get(normalized_observation_label, 0.0)
        ) + float(observation.confidence)
        label_max_confidence[normalized_observation_label] = max(
            float(label_max_confidence.get(normalized_observation_label, 0.0)), float(observation.confidence)
        )
        candidate_evidence = dict(match.attributes.get("label_candidate_evidence", {}))
        self._merge_candidate_evidence(candidate_evidence, observation.label_candidates)
        match.attributes.update(observation.attributes)
        match.attributes["label_evidence"] = label_evidence
        match.attributes["label_score_evidence"] = label_score_evidence
        match.attributes["label_max_confidence"] = label_max_confidence
        match.attributes["label_candidate_evidence"] = candidate_evidence
        if not isinstance(match.attributes.get("label_refinement"), dict):
            winner = max(
                label_evidence,
                key=lambda label: (
                    float(label_max_confidence.get(label, 0.0)),
                    float(label_score_evidence.get(label, 0.0)),
                    int(label_evidence[label]),
                    label,
                ),
            )
            current_label = match.label.casefold()
            if (
                winner != current_label
                and current_label in label_evidence
                and float(label_max_confidence[winner])
                < float(label_max_confidence[current_label]) + self.label_switch_confidence_margin
            ):
                winner = current_label
            recurring_label = max(
                label_evidence,
                key=lambda label: (
                    int(label_evidence[label]),
                    float(label_score_evidence.get(label, 0.0)),
                    float(label_max_confidence.get(label, 0.0)),
                    label,
                ),
            )
            recurring_count = int(label_evidence[recurring_label])
            winner_count = int(label_evidence[winner])
            if (
                recurring_label != winner
                and recurring_count >= 3
                and recurring_count >= self.label_recurrence_count_ratio * winner_count
                and float(label_max_confidence[winner])
                < float(label_max_confidence[recurring_label]) + self.label_high_confidence_override_margin
            ):
                winner = recurring_label
            match.label = winner
            match.canonical_label = winner
            match.confidence = float(label_score_evidence[winner]) / int(label_evidence[winner])
        match.object_version += 1
        match.map_version = observation.map_version or match.map_version
        match.session_id = observation.session_id or match.session_id
        match.model_versions.update(observation.model_versions)
        match.semantic_identities.update(observation.semantic_identities)
        match.deployment_provenance.update(observation.deployment_provenance)
        if observation.embedding is not None:
            if match.embedding is None:
                match.embedding = observation.embedding
            else:
                match.embedding = normalize_embedding((1.0 - alpha) * match.embedding + alpha * observation.embedding)
        return match

    @staticmethod
    def _candidate_evidence(candidates: tuple[tuple[str, float], ...]) -> dict:
        evidence: dict = {}
        SemanticTracker._merge_candidate_evidence(evidence, candidates)
        return evidence

    @staticmethod
    def _merge_candidate_evidence(evidence: dict, candidates: tuple[tuple[str, float], ...]) -> None:
        for label, score in candidates:
            normalized = str(label).strip().casefold()
            if not normalized:
                continue
            current = evidence.get(normalized, {})
            evidence[normalized] = {
                "count": int(current.get("count", 0)) + 1,
                "score_sum": float(current.get("score_sum", 0.0)) + float(score),
                "max_score": max(float(current.get("max_score", 0.0)), float(score)),
            }

    @staticmethod
    def aggregated_label_candidates(
        track: SemanticTrack, limit: int = 5, excluded_labels=()
    ) -> tuple[tuple[str, float], ...]:
        """Return track-level candidates ranked by recurrence and mean confidence."""
        evidence = track.attributes.get("label_candidate_evidence", {})
        excluded = {str(value).strip().casefold() for value in excluded_labels}
        ranked = sorted(
            ((label, values) for label, values in evidence.items() if label.casefold() not in excluded),
            key=lambda item: (
                -int(item[1].get("count", 0)),
                -(float(item[1].get("score_sum", 0.0)) / max(1, int(item[1].get("count", 0)))),
                -float(item[1].get("max_score", 0.0)),
                item[0],
            ),
        )
        return tuple(
            (label, float(values.get("score_sum", 0.0)) / max(1, int(values.get("count", 0))))
            for label, values in ranked[:limit]
        )

    def mark_stale(self, now_ns: int | None = None) -> bool:
        now_ns = time.time_ns() if now_ns is None else now_ns
        changed = False
        for track in self.tracks.values():
            if track.state in {OBSERVED, MOVED} and now_ns - track.last_seen_ns > self.stale_after_ns:
                self.transition(
                    track.object_id,
                    STALE,
                    LifecycleEvidence(freshness_expired=True, details={"now_ns": now_ns}),
                )
                changed = True
        return changed

    def transition(self, object_id: str, new_state: str, evidence: LifecycleEvidence) -> SemanticTrack:
        if new_state not in LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle state: {new_state}")
        track = self.tracks[object_id]
        if new_state not in _ALLOWED_TRANSITIONS[track.state]:
            raise ValueError(f"invalid lifecycle transition: {track.state} -> {new_state}")
        if new_state == OBSERVED and not (evidence.identity_confirmed and evidence.geometry_confirmed):
            raise ValueError("observed transition requires identity and geometry evidence")
        if new_state == MOVED and not (evidence.identity_confirmed and evidence.geometry_confirmed):
            raise ValueError("moved transition requires identity and stable geometry evidence")
        if new_state == STALE and not evidence.freshness_expired:
            raise ValueError("stale transition requires freshness-expiry evidence")
        if new_state == MISSING and not evidence.geometry_confirmed:
            raise ValueError("missing transition requires observed-empty geometry evidence")
        if new_state == LOST and not evidence.search_exhausted:
            raise ValueError("lost transition requires exhausted-search evidence")
        if track.state != new_state:
            track.state = new_state
            track.object_version += 1
        track.lifecycle_evidence = {
            "identity_confirmed": evidence.identity_confirmed,
            "geometry_confirmed": evidence.geometry_confirmed,
            "search_exhausted": evidence.search_exhausted,
            "freshness_expired": evidence.freshness_expired,
            "details": dict(evidence.details),
        }
        return track

    def mark_missing(self, object_id: str, evidence: LifecycleEvidence) -> SemanticTrack:
        return self.transition(object_id, MISSING, evidence)

    def mark_moved(self, object_id: str, position: np.ndarray, evidence: LifecycleEvidence) -> SemanticTrack:
        previous_state = self.tracks[object_id].state
        previous_position = self.tracks[object_id].position.copy()
        track = self.transition(object_id, MOVED, evidence)
        track.position = np.asarray(position, dtype=np.float64)
        if previous_state == MOVED and not np.allclose(previous_position, track.position):
            track.object_version += 1
        return track

    def mark_lost(self, object_id: str, evidence: LifecycleEvidence) -> SemanticTrack:
        return self.transition(object_id, LOST, evidence)

    def _best_match(self, observation: SemanticObservation, excluded_object_ids: set[str]) -> SemanticTrack | None:
        best_track = None
        best_score = float("-inf")
        for track in self.tracks.values():
            if track.object_id in excluded_object_ids:
                continue
            association_labels = {track.label.casefold()}
            refinement = track.attributes.get("label_refinement", {})
            if isinstance(refinement, dict) and refinement.get("previous_label"):
                association_labels.add(str(refinement["previous_label"]).casefold())
            distance = float(np.linalg.norm(track.position - observation.position))
            if distance > self.association_distance_m:
                continue
            track_extent = float(np.linalg.norm(track.size))
            observation_extent = float(np.linalg.norm(observation.size))
            if min(track_extent, observation_extent) > 1e-6:
                size_ratio = max(track_extent, observation_extent) / min(track_extent, observation_extent)
                if size_ratio > self.max_size_ratio:
                    continue
            similarity = cosine_similarity(track.embedding, observation.embedding)
            if observation.label.casefold() not in association_labels and similarity is None:
                continue
            if similarity is not None and similarity < self.embedding_similarity_threshold:
                continue
            distance_score = 1.0 - distance / self.association_distance_m
            score = self.position_weight * distance_score
            if similarity is not None:
                score += (1.0 - self.position_weight) * similarity
            if score > best_score:
                best_track, best_score = track, score
        return best_track
