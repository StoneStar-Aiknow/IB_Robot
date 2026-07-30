import numpy as np
import pytest

from semantic_mapping.association import (
    FROZEN,
    LOST,
    MISSING,
    MOVED,
    OBSERVED,
    STALE,
    LifecycleEvidence,
    SemanticObservation,
    SemanticTrack,
    SemanticTracker,
)


def _observation(position, stamp_ns, embedding):
    return SemanticObservation(
        label="cup",
        confidence=0.8,
        position=np.asarray(position, dtype=np.float64),
        size=np.array([0.08, 0.08, 0.12]),
        point_count=100,
        stamp_ns=stamp_ns,
        embedding=np.asarray(embedding, dtype=np.float32),
    )


def test_nearby_same_identity_updates_persistent_track():
    tracker = SemanticTracker(association_distance_m=0.5, embedding_similarity_threshold=0.7)
    first = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0]))
    second = tracker.update(_observation([0.1, 0.0, 1.0], 2, [0.99, 0.01]))

    assert second.object_id == first.object_id
    assert second.observation_count == 2


def test_siglip_feature_separates_nearby_same_label_objects():
    tracker = SemanticTracker(association_distance_m=0.5, embedding_similarity_threshold=0.7)
    first = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0]))
    second = tracker.update(_observation([0.05, 0.0, 1.0], 2, [0.0, 1.0]))

    assert second.object_id != first.object_id
    assert len(tracker.tracks) == 2


def test_same_frame_detections_cannot_update_one_track_twice():
    tracker = SemanticTracker(association_distance_m=0.5, embedding_similarity_threshold=0.7)
    first = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0]))
    matched_ids = {first.object_id}

    second = tracker.update(
        _observation([0.05, 0.0, 1.0], 1, [1.0, 0.0]),
        excluded_object_ids=matched_ids,
    )

    assert second.object_id != first.object_id
    assert len(tracker.tracks) == 2


def test_track_becomes_inactive_after_stale_timeout():
    tracker = SemanticTracker(stale_after_sec=1.0)
    track = tracker.update(_observation([0.0, 0.0, 1.0], 1_000_000_000, [1.0, 0.0]))

    assert tracker.mark_stale(2_100_000_001)
    assert not track.active
    assert track.state == STALE


def test_frozen_track_requires_identity_and_geometry_to_become_observed():
    tracker = SemanticTracker()
    track = SemanticTrack(
        object_id="offline-object",
        label="cup",
        confidence=0.8,
        position=np.zeros(3),
        size=np.ones(3),
        point_count=10,
        first_seen_ns=1,
        last_seen_ns=1,
        state=FROZEN,
    )
    tracker.add_track(track)

    with pytest.raises(ValueError, match="identity and geometry"):
        tracker.transition("offline-object", OBSERVED, LifecycleEvidence(identity_confirmed=True))
    tracker.transition(
        "offline-object",
        OBSERVED,
        LifecycleEvidence(identity_confirmed=True, geometry_confirmed=True),
    )

    assert track.state == OBSERVED
    assert track.active


def test_missing_moved_and_lost_transitions_require_structured_evidence():
    tracker = SemanticTracker()
    track = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0]))

    tracker.mark_missing(track.object_id, LifecycleEvidence(geometry_confirmed=True))
    assert track.state == MISSING
    with pytest.raises(ValueError, match="stable geometry"):
        tracker.mark_moved(track.object_id, np.ones(3), LifecycleEvidence(identity_confirmed=True))
    tracker.mark_moved(
        track.object_id,
        np.ones(3),
        LifecycleEvidence(identity_confirmed=True, geometry_confirmed=True),
    )
    assert track.state == MOVED
    tracker.transition(
        track.object_id,
        OBSERVED,
        LifecycleEvidence(identity_confirmed=True, geometry_confirmed=True),
    )
    tracker.mark_missing(track.object_id, LifecycleEvidence(geometry_confirmed=True))
    with pytest.raises(ValueError, match="exhausted-search"):
        tracker.mark_lost(track.object_id, LifecycleEvidence())
    tracker.mark_lost(track.object_id, LifecycleEvidence(search_exhausted=True))

    assert track.state == LOST
    assert not track.active


def test_invalid_lifecycle_transition_is_rejected():
    tracker = SemanticTracker()
    track = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0]))

    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        tracker.transition(track.object_id, LOST, LifecycleEvidence(search_exhausted=True))
