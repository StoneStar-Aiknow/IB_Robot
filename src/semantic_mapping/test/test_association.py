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


def _observation(position, stamp_ns, embedding, *, label="cup", confidence=0.8, candidates=(), size=None):
    return SemanticObservation(
        label=label,
        confidence=confidence,
        position=np.asarray(position, dtype=np.float64),
        size=np.array([0.08, 0.08, 0.12]) if size is None else np.asarray(size, dtype=np.float64),
        point_count=100,
        stamp_ns=stamp_ns,
        embedding=np.asarray(embedding, dtype=np.float32),
        label_candidates=candidates,
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


def test_siglip_identity_preserves_track_across_label_disagreement():
    tracker = SemanticTracker(association_distance_m=0.5, embedding_similarity_threshold=0.7)
    first = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0]))
    conflicting = _observation([0.05, 0.0, 1.0], 2, [0.99, 0.01])
    conflicting.label = "mug"

    second = tracker.update(conflicting)

    assert second.object_id == first.object_id
    assert second.attributes["label_evidence"] == {"cup": 1, "mug": 1}


def test_geometry_size_jump_cannot_capture_existing_track():
    tracker = SemanticTracker(
        association_distance_m=0.5,
        embedding_similarity_threshold=0.7,
        max_size_ratio=4.0,
    )
    first = tracker.update(_observation([0.0, 0.0, 0.0], 1, [1.0, 0.0], label="brush", size=[0.05, 0.13, 0.03]))
    floor_patch = tracker.update(_observation([0.1, 0.0, 0.0], 2, [1.0, 0.0], label="table", size=[0.75, 0.43, 0.04]))

    assert floor_patch.object_id != first.object_id
    assert len(tracker.tracks) == 2


def test_track_label_alignment_prefers_high_confidence_over_raw_count():
    """Max-confidence-first election keeps a frequent weak label from beating a stronger label."""
    tracker = SemanticTracker(association_distance_m=0.5, embedding_similarity_threshold=0.7)
    tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0], label="brush", confidence=0.72))
    tracker.update(_observation([0.01, 0.0, 1.0], 2, [1.0, 0.0], label="brush", confidence=0.71))
    track = tracker.update(_observation([0.02, 0.0, 1.0], 3, [1.0, 0.0], label="banana", confidence=0.95))

    assert track.label == "banana"
    assert track.confidence == pytest.approx(0.95)


def test_close_one_shot_confidence_does_not_replace_established_label():
    tracker = SemanticTracker(
        association_distance_m=0.5,
        embedding_similarity_threshold=0.7,
        label_switch_confidence_margin=0.05,
    )
    track = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0], label="chair", confidence=0.891))
    track = tracker.update(_observation([0.01, 0.0, 1.0], 2, [1.0, 0.0], label="paper", confidence=0.900))

    assert track.label == "chair"
    assert track.confidence == pytest.approx(0.891)


def test_recurrent_label_corrects_an_early_high_confidence_outlier():
    tracker = SemanticTracker(association_distance_m=0.5, embedding_similarity_threshold=0.7)
    track = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0], label="paper", confidence=0.93))
    for stamp in range(2, 6):
        track = tracker.update(_observation([0.0, 0.0, 1.0], stamp, [1.0, 0.0], label="cardboard box", confidence=0.90))

    assert track.label == "cardboard box"
    assert track.confidence == pytest.approx(0.90)


def test_large_confidence_advantage_survives_recurrent_noise():
    tracker = SemanticTracker(association_distance_m=0.5, embedding_similarity_threshold=0.7)
    track = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0], label="banana", confidence=0.996))
    for stamp in range(2, 8):
        track = tracker.update(_observation([0.0, 0.0, 1.0], stamp, [1.0, 0.0], label="straw", confidence=0.876))

    assert track.label == "banana"
    assert track.confidence == pytest.approx(0.996)


def test_cloud_refinement_confidence_is_preserved_across_new_observations():
    tracker = SemanticTracker(association_distance_m=0.5, embedding_similarity_threshold=0.7)
    track = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0], label="food", confidence=0.6))
    track.label = "banana"
    track.canonical_label = "banana"
    track.confidence = 0.93
    track.attributes["label_refinement"] = {"source": "cloud_vlm"}

    updated = tracker.update(_observation([0.01, 0.0, 1.0], 2, [1.0, 0.0], label="fruit", confidence=0.7))

    assert updated.label == "banana"
    assert updated.confidence == pytest.approx(0.93)


def test_track_candidates_are_aggregated_across_frames():
    tracker = SemanticTracker(association_distance_m=0.5, embedding_similarity_threshold=0.7)
    tracker.update(
        _observation(
            [0.0, 0.0, 1.0],
            1,
            [1.0, 0.0],
            candidates=(("banana", 0.8), ("food", 0.7), ("brush", 0.9)),
        )
    )
    track = tracker.update(
        _observation(
            [0.02, 0.0, 1.0],
            2,
            [1.0, 0.0],
            candidates=(("banana", 0.9), ("food", 0.6), ("fruit", 0.95)),
        )
    )

    candidates = tracker.aggregated_label_candidates(track)
    assert [label for label, _score in candidates] == ["banana", "food", "fruit", "brush"]
    assert [score for _label, score in candidates] == pytest.approx([0.85, 0.65, 0.95, 0.9])


def test_track_candidate_aggregation_retains_five_hints_for_cloud_review():
    track = SemanticTrack(
        object_id="object-1",
        label="banana",
        confidence=0.8,
        position=np.zeros(3),
        size=np.ones(3),
        point_count=10,
        first_seen_ns=1,
        last_seen_ns=1,
        attributes={
            "label_candidate_evidence": {
                label: {"count": count, "score_sum": score * count, "max_score": score}
                for label, count, score in (
                    ("food", 5, 0.8),
                    ("fruit", 4, 0.8),
                    ("brush", 3, 0.8),
                    ("banana", 2, 0.9),
                    ("sponge", 1, 0.7),
                    ("toy", 1, 0.6),
                )
            }
        },
    )

    assert [label for label, _score in SemanticTracker.aggregated_label_candidates(track)] == [
        "food",
        "fruit",
        "brush",
        "banana",
        "sponge",
    ]
    assert [
        label
        for label, _score in SemanticTracker.aggregated_label_candidates(
            track, excluded_labels=["food", "fruit", "brush"]
        )
    ] == ["banana", "sponge", "toy"]


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


def test_moved_track_becomes_stale_after_timeout():
    tracker = SemanticTracker(stale_after_sec=1.0)
    track = tracker.update(_observation([0.0, 0.0, 1.0], 1_000_000_000, [1.0, 0.0]))
    tracker.mark_moved(
        track.object_id,
        np.asarray([1.0, 0.0, 1.0]),
        LifecycleEvidence(identity_confirmed=True, geometry_confirmed=True),
    )

    assert tracker.mark_stale(2_100_000_001)
    assert track.state == STALE
    assert not track.active


@pytest.mark.parametrize("initial_state", [FROZEN, LOST])
def test_confirmed_movement_can_relocate_inactive_track(initial_state):
    tracker = SemanticTracker()
    track = tracker.update(_observation([0.0, 0.0, 1.0], 1, [1.0, 0.0]))
    track.state = initial_state

    tracker.mark_moved(
        track.object_id,
        np.asarray([1.0, 0.0, 1.0]),
        LifecycleEvidence(identity_confirmed=True, geometry_confirmed=True),
    )

    assert track.state == MOVED
    assert np.allclose(track.position, [1.0, 0.0, 1.0])


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
