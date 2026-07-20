"""Tests for policy_eval pure helpers."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.policy_eval import (  # noqa: E402
    CalibrationStatus,
    ContractContext,
    PredictionRecord,
    StreamRecord,
    build_replay_frames,
    compare_prediction_documents,
    default_plot_dir,
    filter_observations_by_input_features,
    inspect_calibration,
    is_observation_not_ready,
    load_action_mean_std,
    load_prediction_file,
    load_required_input_features,
    make_eval_ticks,
    missing_topics,
    prediction_document,
    read_rosbag_streams,
    rebase_message_headers,
    replay_publisher_qos,
    selected_indices_for_ticks,
    validate_timestamp_compatibility,
    write_prediction_file,
)


class _Spec:
    def __init__(self, topic: str, key: str = "observation.test", *, is_action: bool = False):
        self.key = key
        self.topic = topic
        self.resample_policy = "hold"
        self.asof_tol_ms = 0
        self.ros_type = "std_msgs/msg/String"
        self.stamp_src = "bag"
        self.is_action = is_action


def _prediction(fingerprint: str, action_dim: int, frames):
    def chunk_size(action):
        array = np.asarray(action)
        if array.ndim == 1:
            return 1 if array.size else 0
        if array.ndim == 3 and array.shape[0] == 1:
            return int(array.shape[1])
        return int(array.shape[0]) if array.ndim else 0

    frames = [
        {
            "deployment_fingerprint": "deployment",
            "sample_timestamp_ns": index,
            "chunk_size": chunk_size(frame.get("action")),
            **frame,
        }
        for index, frame in enumerate(frames)
    ]
    successful = sum(frame.get("success") is True and frame.get("action") is not None for frame in frames)
    return {
        "metadata": {
            "contract_fingerprint": fingerprint,
            "action_dim": action_dim,
            "backend": {"name": fingerprint},
            "bag_digest": "bag",
            "timestamp_policy": "header",
            "frame_stride": 1,
            "policy_state_mode": "continuous",
            "replay_timestamp_mode": "historical",
            "policy_bundle_digest": "policy",
            "deployment_fingerprint": "deployment",
            "deployment_fingerprints": ["deployment"],
            "deployment_identity_consistent": True,
            "selected_frame_count": len(frames),
            "planned_frame_count": len(frames),
            "successful_frame_count": successful,
            "complete": successful == len(frames),
        },
        "frames": frames,
    }


def test_missing_topics_reports_required_contract_topics():
    missing = missing_topics(
        [_Spec("/joint_states"), _Spec("/camera/top")], {"/joint_states": "sensor_msgs/msg/JointState"}
    )

    assert missing == ["/camera/top"]


def test_load_required_input_features_from_policy_config(tmp_path):
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "config.json").write_text(
        '{"input_features": {"observation.state": {}, "observation.images.top": {}}}', encoding="utf-8"
    )

    assert load_required_input_features(policy_dir) == ["observation.state", "observation.images.top"]


def test_filter_observations_by_input_features_skips_unused_contract_topics():
    observations = [
        SimpleNamespace(key="observation.state", topic="/joint_states"),
        SimpleNamespace(key="observation.images.top", topic="/camera/top/image_raw"),
        SimpleNamespace(key="observation.images.front", topic="/camera/front/image_raw"),
    ]

    filtered = filter_observations_by_input_features(
        observations,
        ["observation.state", "observation.images.top"],
    )

    assert [spec.topic for spec in filtered] == ["/joint_states", "/camera/top/image_raw"]


def test_inspect_calibration_accepts_multiple_existing_files(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text("{}", encoding="utf-8")
    right.write_text("{}", encoding="utf-8")

    status = inspect_calibration(
        {"ros2_control": {"xacro_args": {"calib_file_left": str(left), "calib_file_right": str(right)}}}
    )

    assert status.status == "available"
    assert status.path == str(left)
    assert status.paths == (str(left), str(right))
    assert status.message == "calibration files exist"


def test_inspect_calibration_reports_missing_files(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text("{}", encoding="utf-8")

    status = inspect_calibration(
        {"ros2_control": {"xacro_args": {"calib_file_left": str(left), "calib_file_right": str(right)}}}
    )

    assert status.status == "missing"
    assert status.path == str(left)
    assert status.paths == (str(left), str(right))
    assert status.message == f"missing calibration files: {right}"


def test_make_eval_ticks_applies_stride_and_limit():
    ticks = make_eval_ticks([[0, 100_000_000, 200_000_000, 300_000_000]], 10.0, frame_limit=2, frame_stride=2)

    assert ticks == [0, 200_000_000]


def test_make_eval_ticks_starts_when_all_streams_have_data():
    # Two streams whose first message timestamps differ by 100 ms (e.g. a USB
    # camera whose header stamp predates joint-state bag time). Ticks must start
    # at the *latest* first-message time so every stream has data at tick 0.
    stream_a = [0, 100_000_000, 200_000_000, 300_000_000]  # starts at 0
    stream_b = [100_000_000, 200_000_000, 300_000_000, 400_000_000]  # starts at 100ms
    ticks = make_eval_ticks([stream_a, stream_b], 10.0)

    assert ticks[0] == 100_000_000
    assert all(t >= 100_000_000 for t in ticks)


def test_selected_indices_for_asof_honors_tolerance():
    selected = selected_indices_for_ticks(
        policy="asof",
        timestamps_ns=[0, 100, 200],
        ticks_ns=[50, 180, 260],
        step_ns=100,
        tol_ns=60,
    )

    assert selected == [0, None, 2]


def test_selected_indices_for_hold_does_not_select_future_frame():
    selected = selected_indices_for_ticks(
        policy="hold",
        timestamps_ns=[100, 200],
        ticks_ns=[0, 50, 100, 150],
        step_ns=100,
        tol_ns=0,
    )

    assert selected == [None, None, 0, 0]


def test_selected_indices_for_drop_does_not_select_future_frame():
    selected = selected_indices_for_ticks(
        policy="drop",
        timestamps_ns=[100, 200],
        ticks_ns=[0, 50, 100],
        step_ns=100,
        tol_ns=0,
    )

    assert selected == [None, None, 0]


def test_build_replay_frames_skips_late_stream_before_first_frame():
    early = StreamRecord(
        spec=_Spec("/early", "observation.early"),
        timestamps_ns=[0, 100],
        messages=["early-0", "early-100"],
    )
    late = StreamRecord(
        spec=_Spec("/late", "observation.late"),
        timestamps_ns=[100, 200],
        messages=["late-100", "late-200"],
    )

    frames = build_replay_frames(
        {"observation.early": early, "observation.late": late},
        [0, 100],
        10.0,
    )

    assert frames[0].observation_messages == {"/early": "early-0"}
    assert [(diag.key, diag.ready, diag.source_timestamp_ns) for diag in frames[0].diagnostics] == [
        ("observation.early", True, 0),
        ("observation.late", False, None),
    ]
    assert frames[1].observation_messages == {"/early": "early-100", "/late": "late-100"}


def test_read_rosbag_streams_keeps_duplicate_state_specs_separate(monkeypatch):
    topics = {
        "/base_controller/odom": "nav_msgs/msg/Odometry",
        "/imu_sensor_broadcaster/imu": "sensor_msgs/msg/Imu",
        "/joint_states": "sensor_msgs/msg/JointState",
    }
    messages = [
        ("/base_controller/odom", "odom-msg", 10),
        ("/imu_sensor_broadcaster/imu", "imu-msg", 20),
        ("/joint_states", "joint-msg", 30),
    ]

    class FakeReader:
        def __init__(self):
            self.index = 0

        def open(self, *_args, **_kwargs):
            return None

        def get_all_topics_and_types(self):
            return [SimpleNamespace(name=topic, type=msg_type) for topic, msg_type in topics.items()]

        def has_next(self):
            return self.index < len(messages)

        def read_next(self):
            item = messages[self.index]
            self.index += 1
            return item

    fake_rosbag2_py = SimpleNamespace(
        SequentialReader=FakeReader,
        StorageOptions=lambda **_kwargs: object(),
        ConverterOptions=lambda **_kwargs: object(),
    )
    fake_serialization = SimpleNamespace(deserialize_message=lambda data, _msg_type: data)
    fake_utilities = SimpleNamespace(get_message=lambda msg_type: msg_type)
    monkeypatch.setitem(sys.modules, "rosbag2_py", fake_rosbag2_py)
    monkeypatch.setitem(sys.modules, "rclpy.serialization", fake_serialization)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.utilities", fake_utilities)

    specs = [_Spec(topic, "observation.state") for topic in topics]
    context = SimpleNamespace(observations=specs, actions=[], policy_path=None)

    _, obs_streams, _ = read_rosbag_streams(
        "/tmp/fake_bag",
        context,
        timestamp_policy="bag",
    )

    assert set(obs_streams) == {
        "observation.state_base_controller_odom",
        "observation.state_imu_sensor_broadcaster_imu",
        "observation.state_joint_states",
    }
    assert obs_streams["observation.state_base_controller_odom"].messages == ["odom-msg"]
    assert obs_streams["observation.state_imu_sensor_broadcaster_imu"].messages == ["imu-msg"]
    assert obs_streams["observation.state_joint_states"].messages == ["joint-msg"]


def test_read_rosbag_streams_keeps_duplicate_action_specs_separate(monkeypatch):
    topics = {
        "/arm_position_controller/commands": "std_msgs/msg/Float64MultiArray",
        "/gripper_position_controller/commands": "std_msgs/msg/Float64MultiArray",
    }
    messages = [
        ("/arm_position_controller/commands", "arm-msg", 10),
        ("/gripper_position_controller/commands", "gripper-msg", 20),
    ]

    class FakeReader:
        def __init__(self):
            self.index = 0

        def open(self, *_args, **_kwargs):
            return None

        def get_all_topics_and_types(self):
            return [SimpleNamespace(name=topic, type=msg_type) for topic, msg_type in topics.items()]

        def has_next(self):
            return self.index < len(messages)

        def read_next(self):
            item = messages[self.index]
            self.index += 1
            return item

    fake_rosbag2_py = SimpleNamespace(
        SequentialReader=FakeReader,
        StorageOptions=lambda **_kwargs: object(),
        ConverterOptions=lambda **_kwargs: object(),
    )
    fake_serialization = SimpleNamespace(deserialize_message=lambda data, _msg_type: data)
    fake_utilities = SimpleNamespace(get_message=lambda msg_type: msg_type)
    monkeypatch.setitem(sys.modules, "rosbag2_py", fake_rosbag2_py)
    monkeypatch.setitem(sys.modules, "rclpy.serialization", fake_serialization)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.utilities", fake_utilities)
    monkeypatch.setattr(
        "dataset_tools.policy_eval._decode_contract_value",
        lambda _ros_type, msg, _spec: [1.0, 2.0] if msg == "arm-msg" else [3.0],
    )

    specs = [_Spec(topic, "action", is_action=True) for topic in topics]
    context = SimpleNamespace(observations=[], actions=specs, policy_path=None)

    _, _, action_streams = read_rosbag_streams(
        "/tmp/fake_bag",
        context,
        timestamp_policy="bag",
        include_actions=True,
    )

    assert set(action_streams) == {
        "action_arm_position_controller_commands",
        "action_gripper_position_controller_commands",
    }
    assert action_streams["action_arm_position_controller_commands"].decoded_values == [[1.0, 2.0]]
    assert action_streams["action_gripper_position_controller_commands"].decoded_values == [[3.0]]

    frames = build_replay_frames({}, [20], 10.0, action_streams=action_streams)

    assert frames[0].label_action == [1.0, 2.0, 3.0]


def test_timestamp_compatibility_rejects_non_header_contract_streams():
    with pytest.raises(ValueError, match="stamp_src='header'"):
        validate_timestamp_compatibility("header", [_Spec("/joint_states")])


def test_timestamp_compatibility_accepts_header_stamped_streams():
    spec = _Spec("/camera/top")
    spec.stamp_src = "header"

    validate_timestamp_compatibility("contract", [spec])


def test_timestamp_compatibility_rejects_bag_timestamp_replay():
    spec = _Spec("/camera/top")
    spec.stamp_src = "header"

    with pytest.raises(ValueError, match="bag/receive timestamps"):
        validate_timestamp_compatibility("bag", [spec])


def test_rebase_message_headers_copies_messages_and_assigns_live_timestamp():
    original = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=34)), data=[1, 2, 3])

    rebased = rebase_message_headers({"/camera/top": original}, 5_000_000_006)

    assert rebased["/camera/top"] is not original
    assert rebased["/camera/top"].header.stamp.sec == 5
    assert rebased["/camera/top"].header.stamp.nanosec == 6
    assert original.header.stamp.sec == 12
    assert original.header.stamp.nanosec == 34


def test_rebase_message_headers_rejects_messages_without_header():
    with pytest.raises(ValueError, match="/joint_states"):
        rebase_message_headers({"/joint_states": SimpleNamespace()}, 1)


def test_observation_not_ready_detection_is_specific():
    assert is_observation_not_ready("observation_not_ready")
    assert not is_observation_not_ready(None)


def test_replay_publisher_qos_forces_reliable_without_mutating_contract_qos():
    contract_qos = {"reliability": "best_effort", "history": "keep_last", "depth": 10}

    replay_qos = replay_publisher_qos(contract_qos)

    assert replay_qos == {"reliability": "reliable", "history": "keep_last", "depth": 1}
    assert contract_qos["reliability"] == "best_effort"
    assert contract_qos["depth"] == 10


def test_compare_prediction_documents_reports_metrics():
    reference = _prediction(
        "abc",
        2,
        [
            {"frame_index": 0, "sample_timestamp_ns": 10, "success": True, "action": [[1.0, 2.0]]},
            {"frame_index": 1, "sample_timestamp_ns": 20, "success": True, "action": [[3.0, 4.0]]},
        ],
    )
    candidate = _prediction(
        "abc",
        2,
        [
            {"frame_index": 0, "sample_timestamp_ns": 10, "success": True, "action": [[2.0, 2.0]]},
            {"frame_index": 1, "sample_timestamp_ns": 20, "success": True, "action": [[5.0, 4.0]]},
        ],
    )

    result = compare_prediction_documents(reference, candidate)

    assert result["matched_frames"] == 2
    assert result["mae"] == pytest.approx(0.75)
    assert result["max_error"] == pytest.approx(2.0)
    assert result["cosine_similarity"] == pytest.approx(37 / (30**0.5 * 49**0.5))
    assert result["mean_frame_cosine_similarity"] == pytest.approx(
        ((6 / (5**0.5 * 8**0.5)) + (31 / (25**0.5 * 41**0.5))) / 2
    )
    assert result["min_frame_cosine_similarity"] == pytest.approx(6 / (5**0.5 * 8**0.5))
    assert result["mean_first_step_cosine_similarity"] == result["mean_frame_cosine_similarity"]
    assert result["min_first_step_cosine_similarity"] == result["min_frame_cosine_similarity"]
    assert result["undefined_frame_cosine_count"] == 0
    assert result["undefined_first_step_cosine_count"] == 0
    assert result["per_action_dim_cosine_similarity"] == pytest.approx([17 / (10**0.5 * 29**0.5), 1.0])
    assert result["per_action_dim_mae"] == pytest.approx([1.5, 0.0])


def test_compare_prediction_documents_reports_undefined_zero_vector_cosines():
    reference = _prediction(
        "abc",
        2,
        [
            {"frame_index": 0, "success": True, "action": [[0.0, 0.0]]},
            {"frame_index": 1, "success": True, "action": [[1.0, 0.0]]},
        ],
    )
    candidate = _prediction(
        "abc",
        2,
        [
            {"frame_index": 0, "success": True, "action": [[0.0, 0.0]]},
            {"frame_index": 1, "success": True, "action": [[-1.0, 0.0]]},
        ],
    )

    result = compare_prediction_documents(reference, candidate)

    assert result["cosine_similarity"] == pytest.approx(-1.0)
    assert result["mean_frame_cosine_similarity"] == pytest.approx(-1.0)
    assert result["min_frame_cosine_similarity"] == pytest.approx(-1.0)
    assert result["undefined_frame_cosine_count"] == 1
    assert result["undefined_first_step_cosine_count"] == 1
    assert result["per_action_dim_cosine_similarity"][0] == pytest.approx(-1.0)
    assert result["per_action_dim_cosine_similarity"][1] is None


def test_compare_prediction_documents_reports_normalized_cosines():
    reference = _prediction(
        "abc",
        2,
        [{"frame_index": 0, "success": True, "action": [[11.0, 18.0], [13.0, 24.0]]}],
    )
    candidate = _prediction(
        "abc",
        2,
        [{"frame_index": 0, "success": True, "action": [[12.0, 18.0], [14.0, 22.0]]}],
    )

    result = compare_prediction_documents(
        reference,
        candidate,
        action_mean=np.array([10.0, 20.0]),
        action_std=np.array([2.0, 4.0]),
    )

    normalized_reference = np.array([[0.5, -0.5], [1.5, 1.0]])
    normalized_candidate = np.array([[1.0, -0.5], [2.0, 0.5]])
    expected = np.dot(normalized_reference.ravel(), normalized_candidate.ravel()) / (
        np.linalg.norm(normalized_reference) * np.linalg.norm(normalized_candidate)
    )
    assert result["normalized_cosine_similarity"] == pytest.approx(expected)
    assert result["normalized_undefined_frame_cosine_count"] == 0


def test_load_action_mean_std_reads_mean_std_unnormalizer(tmp_path):
    safetensors = pytest.importorskip("safetensors.numpy")
    (tmp_path / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {"norm_map": {"ACTION": "MEAN_STD"}},
                        "state_file": "stats.safetensors",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    safetensors.save_file(
        {"action.mean": np.array([1.0, 2.0]), "action.std": np.array([3.0, 4.0])},
        tmp_path / "stats.safetensors",
    )

    mean, std = load_action_mean_std(tmp_path)

    np.testing.assert_array_equal(mean, [1.0, 2.0])
    np.testing.assert_array_equal(std, [3.0, 4.0])


def test_compare_prediction_documents_normalizes_singleton_batch_dimension():
    reference = _prediction(
        "abc",
        2,
        [{"frame_index": 0, "success": True, "action": [[1.0, 2.0], [3.0, 4.0]]}],
    )
    candidate = _prediction(
        "abc",
        2,
        [{"frame_index": 0, "success": True, "action": [[[1.5, 2.0], [3.0, 5.0]]]}],
    )

    result = compare_prediction_documents(reference, candidate)

    assert result["compared_frames"] == 1
    assert result["mismatched_shape_frames"] == []
    assert result["mae"] == pytest.approx(0.375)


def test_compare_prediction_documents_keeps_non_singleton_batches_incompatible():
    reference = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    candidate = _prediction(
        "abc",
        2,
        [{"frame_index": 0, "success": True, "action": [[[1.0, 2.0]], [[1.0, 2.0]]]}],
    )

    with pytest.raises(ValueError, match="non-empty finite numeric 1D/2D arrays"):
        compare_prediction_documents(reference, candidate)


@pytest.mark.parametrize("action", [[float("nan"), 1.0], [float("inf"), 1.0], [float("-inf"), 1.0], []])
def test_compare_prediction_documents_rejects_invalid_one_dimensional_actions(action):
    reference = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [1.0, 2.0]}])
    candidate = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": action}])

    with pytest.raises(ValueError, match="non-empty finite numeric 1D/2D arrays"):
        compare_prediction_documents(reference, candidate)


def test_compare_prediction_documents_rejects_variable_chunk_lengths_by_default():
    frames = [
        {"frame_index": 0, "success": True, "action": [[1.0, 2.0]]},
        {"frame_index": 1, "success": True, "action": [[3.0, 4.0], [5.0, 6.0]]},
    ]
    reference = _prediction("abc", 2, frames)
    candidate = _prediction("abc", 2, frames)

    with pytest.raises(ValueError, match="action chunk shape varies across frames"):
        compare_prediction_documents(reference, candidate)


def test_compare_prediction_documents_rejects_variable_chunk_lengths_even_when_incompatible_allowed():
    frames = [
        {"frame_index": 0, "success": True, "action": [[1.0, 2.0]]},
        {"frame_index": 1, "success": True, "action": [[3.0, 4.0], [5.0, 6.0]]},
    ]

    with pytest.raises(ValueError, match="action chunk shape varies across frames"):
        compare_prediction_documents(
            _prediction("abc", 2, frames),
            _prediction("abc", 2, frames),
            allow_incompatible=True,
        )


def test_compare_prediction_documents_rejects_variable_chunk_length_in_unmatched_frame():
    reference = _prediction(
        "abc",
        2,
        [
            {"frame_index": 0, "success": True, "action": [[1.0, 2.0]]},
            {"frame_index": 1, "success": True, "action": [[3.0, 4.0]]},
        ],
    )
    candidate = _prediction(
        "abc",
        2,
        [
            {"frame_index": 0, "success": True, "action": [[1.0, 2.0]]},
            {"frame_index": 2, "success": True, "action": [[3.0, 4.0], [5.0, 6.0]]},
        ],
    )

    with pytest.raises(ValueError, match="candidate action chunk shape varies across frames"):
        compare_prediction_documents(reference, candidate, allow_incompatible=True)


def test_compare_prediction_documents_rejects_incompatible_action_dim():
    reference = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    candidate = _prediction("abc", 3, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0, 3.0]]}])

    with pytest.raises(ValueError, match="action_dim differs"):
        compare_prediction_documents(reference, candidate)


def test_compare_prediction_documents_rejects_incomplete_run():
    reference = _prediction(
        "abc",
        2,
        [
            {"frame_index": 0, "success": True, "action": [[1.0, 2.0]]},
            {"frame_index": 1, "success": True, "action": [[3.0, 4.0]]},
        ],
    )
    candidate = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    candidate["metadata"]["planned_frame_count"] = 2

    with pytest.raises(ValueError, match="candidate recorded 1/2 planned frames"):
        compare_prediction_documents(reference, candidate)


def test_compare_prediction_documents_rejects_different_successful_frame_sets():
    reference = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    candidate = _prediction("abc", 2, [{"frame_index": 1, "success": True, "action": [[1.0, 2.0]]}])

    with pytest.raises(ValueError, match="successful frame sets differ"):
        compare_prediction_documents(reference, candidate)


@pytest.mark.parametrize(
    "metadata_key",
    [
        "bag_digest",
        "timestamp_policy",
        "frame_stride",
        "policy_state_mode",
        "replay_timestamp_mode",
        "policy_bundle_digest",
    ],
)
def test_compare_prediction_documents_rejects_different_capture_semantics(metadata_key):
    reference = _prediction(
        "abc", 2, [{"frame_index": 0, "sample_timestamp_ns": 10, "success": True, "action": [[1.0, 2.0]]}]
    )
    candidate = _prediction(
        "abc", 2, [{"frame_index": 0, "sample_timestamp_ns": 10, "success": True, "action": [[1.0, 2.0]]}]
    )
    candidate["metadata"][metadata_key] = "different"

    with pytest.raises(ValueError, match=f"{metadata_key} differs"):
        compare_prediction_documents(reference, candidate)


def test_compare_prediction_documents_rejects_timestamp_mismatch_for_frame_index_join():
    reference = _prediction(
        "abc", 2, [{"frame_index": 0, "sample_timestamp_ns": 10, "success": True, "action": [[1.0, 2.0]]}]
    )
    candidate = _prediction(
        "abc", 2, [{"frame_index": 0, "sample_timestamp_ns": 20, "success": True, "action": [[1.0, 2.0]]}]
    )

    with pytest.raises(ValueError, match="sample_timestamp_ns differs"):
        compare_prediction_documents(reference, candidate)


def test_compare_prediction_documents_rejects_any_shape_mismatch_by_default():
    reference = _prediction(
        "abc",
        2,
        [
            {"frame_index": 0, "sample_timestamp_ns": 10, "success": True, "action": [[1.0, 2.0]]},
            {"frame_index": 1, "sample_timestamp_ns": 20, "success": True, "action": [[3.0, 4.0]]},
        ],
    )
    candidate = _prediction(
        "abc",
        2,
        [
            {
                "frame_index": 0,
                "sample_timestamp_ns": 10,
                "success": True,
                "action": [[1.0, 2.0], [3.0, 4.0]],
            },
            {"frame_index": 1, "sample_timestamp_ns": 20, "success": True, "action": [[3.0, 4.0], [5.0, 6.0]]},
        ],
    )

    with pytest.raises(ValueError, match="action shapes differ for frames"):
        compare_prediction_documents(reference, candidate)


def test_compare_prediction_documents_rejects_shape_mismatch_even_when_incompatible_allowed():
    reference = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    candidate = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0], [3.0, 4.0]]}])

    with pytest.raises(ValueError, match="action shapes differ for frames"):
        compare_prediction_documents(reference, candidate, allow_incompatible=True)


@pytest.mark.parametrize("join_key", ["frame_index", "sample_timestamp_ns"])
@pytest.mark.parametrize("duplicate_success", [True, False])
def test_compare_prediction_documents_rejects_duplicate_join_keys_across_all_frames(join_key, duplicate_success):
    frames = [
        {"frame_index": 0, "sample_timestamp_ns": 10, "success": False, "action": None},
        {
            "frame_index": 0 if join_key == "frame_index" else 1,
            "sample_timestamp_ns": 10,
            "success": duplicate_success,
            "action": [[3.0, 4.0]] if duplicate_success else None,
        },
    ]

    with pytest.raises(ValueError, match=f"duplicate {join_key}"):
        compare_prediction_documents(
            _prediction("abc", 2, frames),
            _prediction("abc", 2, frames),
            join_key=join_key,
            allow_incompatible=True,
        )


@pytest.mark.parametrize("action_dim", [None, True, 0, -1, 1.5, "2"])
def test_compare_prediction_documents_requires_positive_integer_action_dim(action_dim):
    document = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    document["metadata"]["action_dim"] = action_dim

    with pytest.raises(ValueError, match="reference action_dim is missing or invalid"):
        compare_prediction_documents(document, document)


def test_compare_prediction_documents_rejects_action_payload_dimension_mismatch():
    document = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0, 3.0]]}])

    with pytest.raises(ValueError, match="reference action payload does not match action_dim 2"):
        compare_prediction_documents(document, document)


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        ([float("nan"), 0.0], [1.0, 1.0]),
        ([float("inf"), 0.0], [1.0, 1.0]),
        ([0.0, 0.0], [float("nan"), 1.0]),
        ([0.0, 0.0], [float("inf"), 1.0]),
        ([0.0, 0.0], [float("-inf"), 1.0]),
        ([0.0, 0.0], [0.0, 1.0]),
    ],
)
def test_compare_prediction_documents_rejects_invalid_normalization_stats(mean, std):
    document = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])

    with pytest.raises(ValueError, match="finite values, and positive std"):
        compare_prediction_documents(document, document, action_mean=np.asarray(mean), action_std=np.asarray(std))


def test_compare_prediction_documents_requires_complete_normalization_stats_pair():
    document = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])

    with pytest.raises(ValueError, match="requires both mean and std"):
        compare_prediction_documents(document, document, action_mean=np.zeros(2))


def test_write_prediction_file_rejects_non_finite_json(tmp_path):
    with pytest.raises(ValueError, match="Out of range float values"):
        write_prediction_file(tmp_path / "prediction.json", {"value": float("nan")})


@pytest.mark.parametrize(
    "payload",
    [
        '{"metadata": {}, "metadata": {}, "frames": []}',
        '{"metadata": {"complete": true, "complete": false}, "frames": []}',
        '{"metadata": {}, "frames": [{"success": true, "success": false}]}',
    ],
)
def test_load_prediction_file_rejects_duplicate_json_keys(tmp_path, payload):
    path = tmp_path / "prediction.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_prediction_file(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_load_prediction_file_rejects_non_finite_json_constants(tmp_path, constant):
    path = tmp_path / "prediction.json"
    path.write_text(f'{{"value": {constant}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_prediction_file(path)


def test_prediction_document_marks_non_finite_action_incomplete():
    context = ContractContext(
        robot_config_path="robot.yaml",
        policy_path="policy",
        required_input_features=[],
        robot_config={},
        contract=SimpleNamespace(name="test"),
        observations=[],
        actions=[],
        fingerprint="contract",
    )
    record = PredictionRecord(
        frame_index=0,
        sample_timestamp_ns=10,
        inference_id="request",
        status="ok",
        success=True,
        message="OK",
        action=[[float("nan"), 1.0]],
    )

    document = prediction_document(
        context=context,
        backend_name="cuda",
        timestamp_policy="header",
        frame_stride=1,
        policy_state_mode="continuous",
        calibration_status=CalibrationStatus(status="ok"),
        records=[record],
        planned_frame_count=1,
        bag_digest="bag",
        policy_bundle_digest="policy",
    )

    assert document["metadata"]["successful_frame_count"] == 0
    assert document["metadata"]["complete"] is False
    assert document["metadata"]["action_dim"] is None


def test_prediction_document_rejects_mixed_deployment_identity():
    context = ContractContext(
        robot_config_path="robot.yaml",
        policy_path="policy",
        required_input_features=[],
        robot_config={},
        contract=SimpleNamespace(name="test"),
        observations=[],
        actions=[],
        fingerprint="contract",
    )
    records = [
        PredictionRecord(
            frame_index=index,
            sample_timestamp_ns=index,
            inference_id=f"request-{index}",
            status="ok",
            success=True,
            message="OK",
            action=[[1.0, 2.0]],
            deployment_fingerprint=f"deployment-{index}",
        )
        for index in range(2)
    ]

    document = prediction_document(
        context=context,
        backend_name="cuda",
        timestamp_policy="header",
        frame_stride=1,
        policy_state_mode="continuous",
        calibration_status=CalibrationStatus(status="ok"),
        records=records,
        planned_frame_count=2,
        bag_digest="bag",
        policy_bundle_digest="policy",
    )

    assert document["metadata"]["complete"] is False
    assert document["metadata"]["deployment_identity_consistent"] is False
    assert document["metadata"]["deployment_fingerprint"] == ""
    assert document["metadata"]["deployment_fingerprints"] == ["deployment-0", "deployment-1"]


def test_prediction_document_marks_mixed_action_dimensions_incomplete():
    context = ContractContext(
        robot_config_path="robot.yaml",
        policy_path="policy",
        required_input_features=[],
        robot_config={},
        contract=SimpleNamespace(name="test"),
        observations=[],
        actions=[],
        fingerprint="contract",
    )
    records = [
        PredictionRecord(
            frame_index=index,
            sample_timestamp_ns=index,
            inference_id=f"request-{index}",
            status="ok",
            success=True,
            message="OK",
            action=action,
            deployment_fingerprint="deployment",
        )
        for index, action in enumerate(([[1.0, 2.0]], [[1.0, 2.0, 3.0]]))
    ]

    document = prediction_document(
        context=context,
        backend_name="cuda",
        timestamp_policy="header",
        frame_stride=1,
        policy_state_mode="continuous",
        calibration_status=CalibrationStatus(status="ok"),
        records=records,
        planned_frame_count=2,
        bag_digest="bag",
        policy_bundle_digest="policy",
    )

    assert document["metadata"]["successful_frame_count"] == 2
    assert document["metadata"]["action_dim"] is None
    assert document["metadata"]["complete"] is False


def test_prediction_document_marks_variable_chunk_shapes_incomplete():
    context = ContractContext(
        robot_config_path="robot.yaml",
        policy_path="policy",
        required_input_features=[],
        robot_config={},
        contract=SimpleNamespace(name="test"),
        observations=[],
        actions=[],
        fingerprint="contract",
    )
    records = [
        PredictionRecord(
            frame_index=index,
            sample_timestamp_ns=index,
            inference_id=f"request-{index}",
            status="ok",
            success=True,
            message="OK",
            action=action,
            deployment_fingerprint="deployment",
        )
        for index, action in enumerate(([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]]))
    ]

    document = prediction_document(
        context=context,
        backend_name="cuda",
        timestamp_policy="header",
        frame_stride=1,
        policy_state_mode="continuous",
        calibration_status=CalibrationStatus(status="ok"),
        records=records,
        planned_frame_count=2,
        bag_digest="bag",
        policy_bundle_digest="policy",
    )

    assert document["metadata"]["action_dim"] == 2
    assert document["metadata"]["complete"] is False


def test_compare_prediction_documents_handles_large_finite_actions_without_overflow():
    reference = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1e300, -1e300]]}])
    candidate = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1e300, -5e299]]}])

    result = compare_prediction_documents(reference, candidate)

    assert math.isfinite(result["mae"])
    assert math.isfinite(result["rmse"])
    assert math.isfinite(result["cosine_similarity"])


def test_compare_prediction_documents_preserves_tiny_vector_cosine():
    reference = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1e-300, -1e-300]]}])
    candidate = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[-1e-300, 1e-300]]}])

    result = compare_prediction_documents(reference, candidate)

    assert result["cosine_similarity"] == pytest.approx(-1.0)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, "1", 2])
def test_compare_prediction_documents_rejects_invalid_frame_chunk_size(chunk_size):
    document = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    document["frames"][0]["chunk_size"] = chunk_size

    with pytest.raises(ValueError, match="chunk_size does not match"):
        compare_prediction_documents(document, document)


def test_compare_prediction_documents_rejects_missing_contract_fingerprint_on_both_sides():
    document = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    del document["metadata"]["contract_fingerprint"]

    with pytest.raises(ValueError, match="contract_fingerprint is missing"):
        compare_prediction_documents(document, document)


def test_compare_prediction_documents_rejects_non_boolean_success():
    document = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    document["frames"][0]["success"] = "false"

    with pytest.raises(ValueError, match="success must be boolean"):
        compare_prediction_documents(document, document)


def test_compare_prediction_documents_rejects_mixed_frame_deployment_identity():
    frames = [
        {"frame_index": 0, "success": True, "action": [[1.0, 2.0]], "deployment_fingerprint": "a"},
        {"frame_index": 1, "success": True, "action": [[3.0, 4.0]], "deployment_fingerprint": "b"},
    ]

    with pytest.raises(ValueError, match="deployment identity is missing or varies"):
        compare_prediction_documents(_prediction("abc", 2, frames), _prediction("abc", 2, frames))


def test_compare_prediction_documents_requires_deployment_identity_proof():
    document = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    del document["metadata"]["deployment_identity_consistent"]
    del document["frames"][0]["deployment_fingerprint"]

    with pytest.raises(ValueError, match="deployment identity proof is missing or inconsistent"):
        compare_prediction_documents(document, document)


def test_default_plot_dir_prefers_compare_output_path():
    assert default_plot_dir("/tmp/gpu_predictions.json", "/tmp/gpu_vs_rknn_metrics.json").as_posix() == (
        "/tmp/gpu_vs_rknn_metrics_plots"
    )


def test_default_plot_dir_falls_back_to_reference_path():
    assert default_plot_dir("/tmp/gpu_predictions.json", "").as_posix() == "/tmp/gpu_predictions_plots"
