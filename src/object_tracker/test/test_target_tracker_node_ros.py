import json
import sqlite3

import numpy as np
import pytest
import rclpy

from ibrobot_msgs.msg import TrackState
from ibrobot_msgs.srv import StartTracking
from object_tracker.motion import MotionEstimate, MotionState
from object_tracker.session import SessionState
from object_tracker.target_tracker_node import TargetTrackerNode


@pytest.fixture(scope="module")
def rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def _write_database(path, objects):
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE semantic_objects ("
        "object_id TEXT PRIMARY KEY, label TEXT, confidence REAL, position_json TEXT, "
        "size_json TEXT, point_count INTEGER, first_seen_ns INTEGER, last_seen_ns INTEGER, "
        "observation_count INTEGER, embedding BLOB, embedding_size INTEGER, state TEXT, "
        "map_version INTEGER, session_id TEXT, object_version INTEGER, model_versions_json TEXT, "
        "semantic_identities_json TEXT, deployment_provenance_json TEXT, lifecycle_evidence_json TEXT, "
        "attributes_json TEXT)"
    )
    for object_id, label, position, size, count, state in objects:
        connection.execute(
            "INSERT INTO semantic_objects (object_id, label, position_json, size_json, observation_count, state) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (object_id, label, json.dumps(position), json.dumps(size), count, state),
        )
    connection.commit()
    connection.close()


def _make_node(tmp_path, objects):
    database = tmp_path / "semantic_map.sqlite3"
    _write_database(str(database), objects)
    node = TargetTrackerNode()
    node.set_parameters(
        [
            rclpy.parameter.Parameter("semantic_database_path", value=str(database)),
        ]
    )
    return node


def _start_request(object_id="", query_text=""):
    request = StartTracking.Request()
    request.object_id = object_id
    request.query_text = query_text
    request.stand_off_distance_m = 0.8
    return request


def test_start_tracking_resolves_query_text(rclpy_context, tmp_path):
    node = _make_node(
        tmp_path,
        [
            ("obj-banana", "banana", [-1.5, -2.4, 0.0], [0.08, 0.08, 0.15], 103, "observed"),
            ("obj-table", "table", [0.5, 0.2, 0.0], [0.7, 0.7, 0.05], 2, "observed"),
        ],
    )
    try:
        response = node._start_tracking(_start_request(query_text="banana"), StartTracking.Response())

        assert response.success
        assert response.resolved_object_id == "obj-banana"
        assert response.initial_state == TrackState.ACQUIRING
        assert node._pending_target is not None
        assert node._pending_target["label"] == "banana"
        assert np.allclose(node._pending_target["position_map"], [-1.5, -2.4, 0.0])
    finally:
        node.destroy_node()


def test_start_tracking_rejects_unknown_target(rclpy_context, tmp_path):
    node = _make_node(tmp_path, [("obj-banana", "banana", [-1.5, -2.4, 0.0], [0.08, 0.08, 0.15], 5, "observed")])
    try:
        response = node._start_tracking(_start_request(query_text="cup"), StartTracking.Response())

        assert not response.success
        assert node._pipeline is None
    finally:
        node.destroy_node()


def test_second_start_rejected_while_active(rclpy_context, tmp_path):
    node = _make_node(tmp_path, [("obj-banana", "banana", [-1.5, -2.4, 0.0], [0.08, 0.08, 0.15], 5, "observed")])
    try:
        first = node._start_tracking(_start_request(query_text="banana"), StartTracking.Response())
        assert first.success

        second = node._start_tracking(_start_tracking := _start_request(query_text="table"), StartTracking.Response())
        assert not second.success
        assert "already exists" in second.message
    finally:
        node.destroy_node()


def test_stop_tracking_terminates_session(rclpy_context, tmp_path):
    node = _make_node(tmp_path, [("obj-banana", "banana", [-1.5, -2.4, 0.0], [0.08, 0.08, 0.15], 5, "observed")])
    try:
        started = node._start_tracking(_start_request(query_text="banana"), StartTracking.Response())
        assert started.success

        from ibrobot_msgs.srv import StopTracking

        request = StopTracking.Request()
        request.session_id = started.session_id
        response = node._stop_tracking(request, StopTracking.Response())

        assert response.success
        assert response.final_state == TrackState.STOPPED
    finally:
        node.destroy_node()


def test_apply_motion_estimate_flags_unknown_motion(rclpy_context, tmp_path):
    state = TrackState()
    state.actionable = True
    estimate = MotionEstimate(
        state=MotionState.UNKNOWN,
        displacement_m=0.0,
        speed_mps=0.0,
        threshold_m=0.08,
        ego_motion_active=False,
        reason="insufficient_samples",
    )

    result = TargetTrackerNode.apply_motion_estimate(state, estimate)

    assert result.motion_state == TrackState.MOTION_UNKNOWN
    assert not result.actionable
    assert result.state_reason == "insufficient_samples"


def test_motion_delegation_uses_pipeline_classifier(rclpy_context, tmp_path):
    node = _make_node(tmp_path, [("obj-banana", "banana", [-1.5, -2.4, 0.0], [0.08, 0.08, 0.15], 5, "observed")])
    try:
        estimate = node.update_target_motion(
            stamp_s=1.0,
            position_odom=(1.0, 2.0),
            position_covariance=np.diag([0.001, 0.001]),
        )
        assert estimate.state is MotionState.UNKNOWN

        node._start_tracking(_start_request(query_text="banana"), StartTracking.Response())
        assert node._pipeline is not None
        estimate = node.update_target_motion(
            stamp_s=2.0,
            position_odom=(1.05, 2.0),
            position_covariance=np.diag([0.001, 0.001]),
        )
        assert node._pipeline.motion is not None
    finally:
        node.destroy_node()


def test_session_module_states_match_node_mapping():
    from object_tracker.target_tracker_node import _STATE_VALUES

    assert _STATE_VALUES[SessionState.TRACKING] == TrackState.TRACKING
    assert _STATE_VALUES[SessionState.LOST] == TrackState.LOST
