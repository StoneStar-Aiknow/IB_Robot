from types import SimpleNamespace

import numpy as np

from semantic_mapping.association import LifecycleEvidence, SemanticObservation, SemanticTracker
from semantic_mapping.manipulation_handoff import handoff_to_manipulation
from semantic_mapping.online_lifecycle import OnlineLifecycleCoordinator
from semantic_mapping.target_resolution import resolve_target
from semantic_mapping.target_watch import TargetWatch


def _track(state="observed"):
    tracker = SemanticTracker(embedding_similarity_threshold=0.8)
    track = tracker.update(
        SemanticObservation(
            label="cup",
            confidence=0.9,
            position=np.asarray([2.0, 0.0, 0.5]),
            size=np.ones(3),
            point_count=10,
            stamp_ns=1,
            embedding=np.asarray([1.0, 0.0]),
        )
    )
    track.state = state
    return tracker, track


def test_object_centroid_is_never_emitted_as_navigation_staging_goal():
    _, track = _track()
    sent_goals = []
    result = resolve_target(track, np.zeros(3), 0.8, lambda candidate: (True, ""))
    if result.ready:
        sent_goals.append(result.staging.position)

    assert len(sent_goals) == 1
    assert not np.allclose(sent_goals[0], track.position)


def test_invalid_states_cannot_start_confirmation_or_grasp():
    for state in ("stale", "missing", "lost"):
        _, track = _track(state)
        calls = []
        result = handoff_to_manipulation(
            track,
            lambda item, calls=calls: calls.append("confirm"),
            lambda confirmation, calls=calls: calls.append("plan"),
        )
        assert not result.success
        assert calls == []


def test_manipulation_handoff_confirms_fresh_target_before_grasp_planning():
    _, track = _track()
    calls = []

    def confirm(item):
        calls.append("confirm")
        return SimpleNamespace(success=True, message="", detections=["fresh"])

    def plan(confirmation):
        calls.append("plan")
        return SimpleNamespace(success=True, message="", grasps=["candidate"])

    result = handoff_to_manipulation(track, confirm, plan)

    assert result.success
    assert calls == ["confirm", "plan"]


def test_target_watch_replans_on_stable_move_and_becomes_lost_after_bounded_search():
    tracker, track = _track()
    lifecycle = OnlineLifecycleCoordinator(tracker, move_confirmations=2)
    watch = TargetWatch(lifecycle, track.object_id, max_attempts=2)
    assert watch.observe(np.asarray([3.0, 0.0, 0.5]), np.asarray([1.0, 0.0])).outcome == "continue"
    assert watch.observe(np.asarray([3.02, 0.0, 0.5]), np.asarray([1.0, 0.0])).outcome == "replan"

    tracker.transition(
        track.object_id,
        "observed",
        LifecycleEvidence(identity_confirmed=True, geometry_confirmed=True),
    )
    lifecycle.mark_expected_region_empty(track.object_id)
    lost_watch = TargetWatch(lifecycle, track.object_id, max_attempts=2)
    assert lost_watch.search_failed().outcome == "continue"
    assert lost_watch.search_failed().outcome == "lost"
