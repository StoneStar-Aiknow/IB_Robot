"""Tests for policy_eval pure helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.policy_eval import (  # noqa: E402
    StreamRecord,
    build_replay_frames,
    compare_prediction_documents,
    default_plot_dir,
    filter_observations_by_input_features,
    inspect_calibration,
    load_required_input_features,
    make_eval_ticks,
    missing_topics,
    read_rosbag_streams,
    selected_indices_for_ticks,
    validate_timestamp_compatibility,
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
    return {
        "metadata": {
            "contract_fingerprint": fingerprint,
            "action_dim": action_dim,
            "backend": {"name": fingerprint},
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
        "robot_config.contract_utils.decode_value",
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


def test_timestamp_compatibility_rejects_receive_time_policy_node():
    with pytest.raises(ValueError, match="use_header_time=true"):
        validate_timestamp_compatibility("header", False)


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
    assert result["per_action_dim_mae"] == pytest.approx([1.5, 0.0])


def test_compare_prediction_documents_rejects_incompatible_action_dim():
    reference = _prediction("abc", 2, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0]]}])
    candidate = _prediction("abc", 3, [{"frame_index": 0, "success": True, "action": [[1.0, 2.0, 3.0]]}])

    with pytest.raises(ValueError, match="action_dim differs"):
        compare_prediction_documents(reference, candidate)


def test_default_plot_dir_prefers_compare_output_path():
    assert default_plot_dir("/tmp/gpu_predictions.json", "/tmp/gpu_vs_rknn_metrics.json").as_posix() == (
        "/tmp/gpu_vs_rknn_metrics_plots"
    )


def test_default_plot_dir_falls_back_to_reference_path():
    assert default_plot_dir("/tmp/gpu_predictions.json", "").as_posix() == "/tmp/gpu_predictions_plots"
