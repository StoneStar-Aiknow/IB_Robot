from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Pose, PoseWithCovariance, PoseWithCovarianceStamped
from ibrobot_msgs.msg import SemanticObject3D
from tf2_ros import TransformException

from semantic_mapping.association import SemanticTrack
from semantic_mapping.semantic_mapping_node import SemanticMappingNode
from semantic_mapping.standoff_query import compute_standoff_pose, object_position_from_msg
from semantic_mapping.target_resolution import resolve_target


def _track(state="observed"):
    return SemanticTrack(
        object_id="object",
        label="cup",
        confidence=0.9,
        position=np.asarray([2.0, 0.0, 0.5]),
        size=np.asarray([0.2, 0.2, 0.2]),
        point_count=10,
        first_seen_ns=1,
        last_seen_ns=2,
        state=state,
    )


def test_resolver_returns_distinct_stand_off_pose_facing_object():
    resolution = resolve_target(
        _track(),
        np.zeros(3),
        0.8,
        lambda candidate: (True, ""),
    )

    assert resolution.ready
    assert not np.allclose(resolution.staging.position, resolution.object.position)
    assert np.linalg.norm(resolution.staging.position[:2] - resolution.object.position[:2]) == 0.8


def test_resolver_checks_candidates_and_rejects_invalid_lifecycle_states():
    checked = []

    def checker(candidate):
        checked.append(candidate)
        return (len(checked) == 2, "blocked")

    assert resolve_target(_track(), np.zeros(3), 0.8, checker).ready
    assert len(checked) == 2
    for state in ("stale", "missing", "lost"):
        checked.clear()
        result = resolve_target(_track(state), np.zeros(3), 0.8, checker)
        assert not result.ready
        assert checked == []


def test_compute_standoff_pose_returns_map_xy_and_degrees():
    x, y, theta = compute_standoff_pose(
        np.asarray([2.0, 0.0, 0.0]),
        np.zeros(3),
        0.2,
    )

    assert np.isclose(x, 1.8)
    assert np.isclose(y, 0.0)
    assert np.isclose(theta, 0.0)


def test_robot_position_uses_live_global_to_base_tf():
    class FakeTFBuffer:
        def lookup_transform(self, target, source, _time, *, timeout):
            assert target == "map"
            assert source == "base_link"
            assert timeout == "timeout"
            return SimpleNamespace(transform=SimpleNamespace(translation=SimpleNamespace(x=1.2, y=-0.4, z=0.1)))

    node = SimpleNamespace(
        global_frame="map",
        tf_timeout="timeout",
        _tf_buffer=FakeTFBuffer(),
        get_parameter=lambda name: SimpleNamespace(value="base_link") if name == "base_frame" else None,
    )

    position = SemanticMappingNode._robot_position_from_tf(node)

    assert np.allclose(position, [1.2, -0.4, 0.1])


def test_robot_position_fails_closed_when_tf_is_unavailable():
    class MissingTFBuffer:
        def lookup_transform(self, _target, _source, _time, *, timeout):
            raise TransformException(f"missing transform after {timeout}")

    node = SimpleNamespace(
        global_frame="map",
        tf_timeout="timeout",
        _tf_buffer=MissingTFBuffer(),
        get_parameter=lambda name: SimpleNamespace(value="base_link") if name == "base_frame" else None,
    )

    with pytest.raises(TransformException, match="missing transform"):
        SemanticMappingNode._robot_position_from_tf(node)


def test_object_position_from_msg_traverses_three_pose_levels():
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = 0.77, -1.12, -0.004
    with_covariance = PoseWithCovariance(pose=pose)
    stamped = PoseWithCovarianceStamped(pose=with_covariance)
    message = SemanticObject3D(pose=stamped)

    position = object_position_from_msg(message)

    assert np.allclose(position, [0.77, -1.12, -0.004])
