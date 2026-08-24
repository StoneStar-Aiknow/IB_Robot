import numpy as np

from semantic_mapping.association import LOST, MISSING, MOVED, SemanticObservation, SemanticTracker
from semantic_mapping.online_lifecycle import OnlineLifecycleCoordinator


def _tracker():
    tracker = SemanticTracker(association_distance_m=0.4, embedding_similarity_threshold=0.8)
    track = tracker.update(
        SemanticObservation(
            label="cup",
            confidence=0.9,
            position=np.zeros(3),
            size=np.ones(3),
            point_count=10,
            stamp_ns=1,
            embedding=np.asarray([1.0, 0.0]),
        )
    )
    return tracker, track


def test_remote_identity_requires_stable_geometry_before_moved_transition():
    tracker, track = _tracker()
    coordinator = OnlineLifecycleCoordinator(tracker, move_confirmations=2)

    assert not coordinator.observe_remote_identity(track.object_id, np.asarray([1.0, 0.0, 0.0]), np.asarray([1.0, 0.0]))
    assert coordinator.observe_remote_identity(track.object_id, np.asarray([1.02, 0.0, 0.0]), np.asarray([1.0, 0.0]))

    assert track.state == MOVED
    assert np.allclose(track.position, [1.01, 0.0, 0.0])


def test_embedding_alone_cannot_move_object_with_conflicting_identity():
    tracker, track = _tracker()
    coordinator = OnlineLifecycleCoordinator(tracker, move_confirmations=1)

    assert not coordinator.observe_remote_identity(track.object_id, np.ones(3), np.asarray([0.0, 1.0]))
    assert np.allclose(track.position, np.zeros(3))


def test_known_tracker_identity_requires_stable_geometry_and_supports_repeated_moves():
    tracker, track = _tracker()
    coordinator = OnlineLifecycleCoordinator(tracker, move_distance_m=0.4, move_confirmations=2)

    assert not coordinator.observe_tracked_identity("unknown", np.ones(3), stamp_ns=1, session_id="first")
    assert not coordinator.observe_tracked_identity(
        track.object_id, np.asarray([1.0, 0.0, 0.0]), stamp_ns=1, session_id="first"
    )
    assert coordinator.observe_tracked_identity(
        track.object_id, np.asarray([1.02, 0.0, 0.0]), stamp_ns=2, session_id="first"
    )
    first_version = track.object_version
    assert track.state == MOVED
    assert track.lifecycle_evidence["details"]["identity_source"] == "track_state"

    assert not coordinator.observe_tracked_identity(
        track.object_id, np.asarray([2.0, 0.0, 0.0]), stamp_ns=3, session_id="first"
    )
    assert coordinator.observe_tracked_identity(
        track.object_id, np.asarray([2.02, 0.0, 0.0]), stamp_ns=4, session_id="first"
    )
    assert track.object_version == first_version + 1
    assert np.allclose(track.position, [2.01, 0.0, 0.0])


def test_tracked_identity_confirmation_does_not_span_sessions_or_time_gaps():
    tracker, track = _tracker()
    coordinator = OnlineLifecycleCoordinator(tracker, move_confirmations=2, move_confirmation_max_gap_sec=1.0)

    assert not coordinator.observe_tracked_identity(
        track.object_id, np.asarray([1.0, 0.0, 0.0]), stamp_ns=1, session_id="first"
    )
    assert not coordinator.observe_tracked_identity(
        track.object_id, np.asarray([1.01, 0.0, 0.0]), stamp_ns=2, session_id="second"
    )
    assert not coordinator.observe_tracked_identity(
        track.object_id, np.asarray([1.02, 0.0, 0.0]), stamp_ns=2_000_000_003, session_id="second"
    )
    assert coordinator.observe_tracked_identity(
        track.object_id, np.asarray([1.03, 0.0, 0.0]), stamp_ns=2_000_000_004, session_id="second"
    )


def test_missing_and_bounded_watch_exhaustion_transition_to_lost():
    tracker, track = _tracker()
    coordinator = OnlineLifecycleCoordinator(tracker)
    coordinator.mark_expected_region_empty(track.object_id)

    assert track.state == MISSING
    coordinator.begin_watch(track.object_id)
    assert not coordinator.record_search_failure(track.object_id, max_attempts=2)
    assert coordinator.record_search_failure(track.object_id, max_attempts=2)
    assert track.state == LOST
