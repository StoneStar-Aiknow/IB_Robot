import threading
from types import SimpleNamespace

import numpy as np
from geometry_msgs.msg import TransformStamped

from ibrobot_msgs.msg import TrackState
from semantic_mapping.association import MOVED, OBSERVED, STALE, SemanticObservation, SemanticTracker
from semantic_mapping.online_lifecycle import OnlineLifecycleCoordinator
from semantic_mapping.pipeline import SerializedCommitter
from semantic_mapping.semantic_mapping_node import SemanticMappingNode


class _Database:
    def __init__(self):
        self.upserts = []

    def upsert(self, track, *_args):
        self.upserts.append((track.object_id, track.object_version, track.position.copy()))


def _node(track, now_ns=10_000_000_000):
    tracker = SemanticTracker(association_distance_m=0.4)
    tracker.add_track(track)
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = "odom"
    transform.transform.translation.x = 1.0
    transform.transform.translation.y = 2.0
    transform.transform.rotation.w = 1.0
    parameters = {
        "track_state_updates_enabled": True,
        "track_state_max_age_sec": 1.0,
        "track_state_max_covariance_m2": 0.25,
        "track_state_frame": "odom",
        "track_state_persist_interval_sec": 1.0,
    }
    published = []
    return SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now_ns)),
        _tf_buffer=SimpleNamespace(lookup_transform=lambda *_args, **_kwargs: transform),
        global_frame="map",
        tf_timeout=None,
        _lifecycle=OnlineLifecycleCoordinator(tracker, move_distance_m=0.4, move_confirmations=2),
        _track_state_persisted_ns={},
        _track_state_watermarks_ns={},
        _live_seen_ns={},
        _track_state_session_ids={},
        _retired_track_state_sessions={},
        _semantic_commit_watermark_ns=0,
        _tracker=tracker,
        _state_lock=threading.RLock(),
        _database=_Database(),
        _committer=SerializedCommitter(),
        _publish_map=published.append,
        get_logger=lambda: SimpleNamespace(warn=lambda _message: None, error=lambda _message: None),
        _manifest=SimpleNamespace(geometry_map_hash="map-hash"),
        _active_geometry_map_hash="map-hash",
        _localization_ready=True,
        _authoritative_map_odom=True,
        _cloud_map_ready=True,
        _run_admission_open=True,
        published=published,
    )


def _track_state(object_id, x, stamp_ns=9_500_000_000):
    state = TrackState()
    state.header.frame_id = "odom"
    state.header.stamp.sec = stamp_ns // 1_000_000_000
    state.header.stamp.nanosec = stamp_ns % 1_000_000_000
    state.object_id = object_id
    state.session_id = "tracking-session"
    state.lifecycle_state = TrackState.TRACKING
    state.motion_state = TrackState.MOVING
    state.measured = True
    state.actionable = True
    state.confidence = 0.9
    state.pose.pose.position.x = x
    state.pose.pose.orientation.w = 1.0
    for index in (0, 7, 14):
        state.pose.covariance[index] = 0.01
    return state


def test_stable_known_track_state_updates_database_and_publishes_map():
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.asarray([0.0, 0.0, 0.7]),
            size=np.ones(3),
            point_count=20,
            stamp_ns=1,
        )
    )
    node = _node(track)

    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.0))
    assert node._database.upserts == []

    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.02, stamp_ns=9_600_000_000))

    assert track.state == MOVED
    assert np.allclose(track.position, [2.01, 2.0, 0.7])
    assert len(node._database.upserts) == 1
    assert node.published == [9_600_000_000]
    assert track.attributes["tracked_position_update"]["session_id"] == "tracking-session"


def test_stationary_track_state_refreshes_stale_object_without_changing_height():
    tracker = SemanticTracker(stale_after_sec=1.0)
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.asarray([2.0, 2.0, 0.7]),
            size=np.ones(3),
            point_count=20,
            stamp_ns=1,
        )
    )
    assert tracker.mark_stale(2_000_000_000)
    assert track.state == STALE
    node = _node(track)
    state = _track_state(track.object_id, 1.0)
    state.motion_state = TrackState.STATIONARY

    SemanticMappingNode._track_state_callback(node, state)

    assert track.state == OBSERVED
    assert np.allclose(track.position, [2.0, 2.0, 0.7])
    assert track.last_seen_ns == 9_500_000_000
    assert len(node._database.upserts) == 1


def test_unknown_or_stale_track_state_cannot_mutate_semantic_map():
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.zeros(3),
            size=np.ones(3),
            point_count=20,
            stamp_ns=1,
        )
    )
    node = _node(track)

    SemanticMappingNode._track_state_callback(node, _track_state("unknown", 1.0))
    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.0, stamp_ns=1))

    assert node._database.upserts == []
    assert node.published == []
    assert np.allclose(track.position, np.zeros(3))


def test_track_state_cannot_write_when_map_identity_admission_is_closed():
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.zeros(3),
            size=np.ones(3),
            point_count=20,
            stamp_ns=1,
        )
    )
    node = _node(track)
    node._active_geometry_map_hash = "other-map"

    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.0))
    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.02, stamp_ns=9_600_000_000))

    assert node._database.upserts == []
    assert np.allclose(track.position, np.zeros(3))


def test_track_state_cannot_roll_back_newer_semantic_observation():
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.asarray([4.0, 0.0, 0.7]),
            size=np.ones(3),
            point_count=20,
            stamp_ns=9_700_000_000,
        )
    )
    node = _node(track)
    node._live_seen_ns[track.object_id] = 9_700_000_000

    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.0))

    assert node._database.upserts == []
    assert np.allclose(track.position, [4.0, 0.0, 0.7])


def test_loaded_offline_map_stamp_does_not_block_fresh_tracking_session():
    """A map built on a foreign clock must accept a new session's track states."""
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.zeros(3),
            size=np.ones(3),
            point_count=20,
            stamp_ns=20_000_000_000,
        )
    )
    assert track.last_seen_ns == 20_000_000_000
    node = _node(track)

    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.0))
    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.02, stamp_ns=9_600_000_000))

    assert track.state == MOVED
    assert np.allclose(track.position, [2.01, 2.0, 0.0])
    assert len(node._database.upserts) == 1


def test_track_state_confirmation_rejects_retired_session_samples():
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.zeros(3),
            size=np.ones(3),
            point_count=20,
            stamp_ns=1,
        )
    )
    node = _node(track)
    first = _track_state(track.object_id, 1.0, stamp_ns=9_100_000_000)
    second = _track_state(track.object_id, 1.0, stamp_ns=9_200_000_000)
    second.session_id = "second-session"
    delayed_first = _track_state(track.object_id, 1.01, stamp_ns=9_300_000_000)
    second_confirmation = _track_state(track.object_id, 1.02, stamp_ns=9_400_000_000)
    second_confirmation.session_id = "second-session"

    for state in (first, second, delayed_first, second_confirmation):
        SemanticMappingNode._track_state_callback(node, state)

    assert track.state == MOVED
    assert track.lifecycle_evidence["details"]["tracking_session_id"] == "second-session"
    assert len(node._database.upserts) == 1


def test_stationary_observed_track_initializes_persistence_watermark():
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.asarray([2.0, 2.0, 0.7]),
            size=np.ones(3),
            point_count=20,
            stamp_ns=1,
        )
    )
    node = _node(track)
    state = _track_state(track.object_id, 1.0)
    state.motion_state = TrackState.STATIONARY

    SemanticMappingNode._track_state_callback(node, state)

    assert len(node._database.upserts) == 1
    assert node._track_state_persisted_ns[track.object_id] == 9_500_000_000


def test_non_odom_track_state_is_rejected():
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.zeros(3),
            size=np.ones(3),
            point_count=20,
            stamp_ns=1,
        )
    )
    node = _node(track)
    state = _track_state(track.object_id, 1.0)
    state.header.frame_id = "camera_link"

    SemanticMappingNode._track_state_callback(node, state)

    assert node._database.upserts == []


def test_terminal_state_blocks_delayed_tracking_samples():
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.zeros(3),
            size=np.ones(3),
            point_count=20,
            stamp_ns=1,
        )
    )
    node = _node(track)
    stopped = _track_state(track.object_id, 1.0, stamp_ns=9_400_000_000)
    stopped.lifecycle_state = TrackState.STOPPED

    SemanticMappingNode._track_state_callback(node, stopped)
    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.0, stamp_ns=9_200_000_000))
    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.02, stamp_ns=9_300_000_000))

    assert node._database.upserts == []
    assert np.allclose(track.position, np.zeros(3))


def test_first_stationary_refresh_persists_under_simulated_time():
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.asarray([2.0, 2.0, 0.7]),
            size=np.ones(3),
            point_count=20,
            stamp_ns=1,
        )
    )
    node = _node(track, now_ns=500_000_000)
    state = _track_state(track.object_id, 1.0, stamp_ns=400_000_000)
    state.motion_state = TrackState.STATIONARY

    SemanticMappingNode._track_state_callback(node, state)

    assert len(node._database.upserts) == 1
    assert node._track_state_persisted_ns[track.object_id] == 400_000_000


def test_delayed_semantic_frame_cannot_rollback_tracked_position():
    tracker = SemanticTracker()
    track = tracker.update(
        SemanticObservation(
            label="banana",
            confidence=0.9,
            position=np.zeros(3),
            size=np.ones(3),
            point_count=20,
            stamp_ns=1,
        )
    )
    node = _node(track)
    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.0, stamp_ns=9_500_000_000))
    SemanticMappingNode._track_state_callback(node, _track_state(track.object_id, 1.02, stamp_ns=9_600_000_000))
    moved_position = track.position.copy()
    delayed = SemanticObservation(
        label="banana",
        confidence=0.9,
        position=np.zeros(3),
        size=np.ones(3),
        point_count=20,
        stamp_ns=9_550_000_000,
    )

    result = SemanticMappingNode._commit_semantic_observation(node, delayed, set())

    assert result is None
    assert len(node._tracker.tracks) == 1
    assert np.allclose(track.position, moved_position)
