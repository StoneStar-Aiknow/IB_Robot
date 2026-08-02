"""Tests for bag_to_lerobot helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.bag_to_lerobot import (  # noqa: E402
    IntegrityReport,
    _build_feature_conversion_table,
    _clean_float_array,
    _dataset_feature_names_for_spec,
    _estimate_stream_rate_hz,
    _log_image_stream_diagnostics,
    _merge_integrity_report,
    _plan_streams,
    _resolve_video_codec,
    _selected_indices_for_ticks,
    discover_video_adapters,
    export_bags_to_lerobot,
)
from robot_config.utils import resolve_calibration_source_specs_from_config  # noqa: E402


def test_resolve_video_codec_prefers_h264_in_auto_mode(monkeypatch):
    import av

    class DummyCodec:
        def __init__(self, is_encoder: bool):
            self.is_encoder = is_encoder

    def fake_codec(name: str, mode: str):
        assert mode == "w"
        return DummyCodec(is_encoder=name == "h264")

    monkeypatch.setattr(av.codec, "Codec", fake_codec)

    assert _resolve_video_codec("auto") == "h264"


def test_resolve_video_codec_falls_back_to_av1_when_h264_missing(monkeypatch):
    import av

    class DummyCodec:
        def __init__(self, is_encoder: bool):
            self.is_encoder = is_encoder

    def fake_codec(name: str, mode: str):
        assert mode == "w"
        if name == "h264":
            raise ValueError("missing")
        return DummyCodec(is_encoder=name == "libsvtav1")

    monkeypatch.setattr(av.codec, "Codec", fake_codec)

    assert _resolve_video_codec("auto") == "libsvtav1"


def test_estimate_stream_rate_hz_uses_timestamp_span():
    ts = [0, 33_333_333, 66_666_666, 100_000_000]

    assert abs(_estimate_stream_rate_hz(ts) - 30.0) < 0.05


def test_selected_indices_for_ticks_exposes_hold_duplicates_from_phase_offset():
    ts = np.array([0, 34, 68], dtype=np.int64)
    ticks = np.array([0, 33, 66], dtype=np.int64)

    selected = _selected_indices_for_ticks(
        policy="hold",
        ts_ns=ts,
        ticks_ns=ticks,
        step_ns=33,
        tol_ns=0,
    )

    assert selected.tolist() == [0, 0, 1]


def test_dataset_feature_names_for_current_preserve_contract_names():
    class Spec:
        key = "observation.current"
        names = ["current.1", "current.2"]

    assert _dataset_feature_names_for_spec(Spec()) == ["current.1", "current.2"]


def test_plan_streams_uses_external_video_instead_of_rosbag_topic():
    class Spec:
        key = "observation.images.top"
        topic = "/camera/top/image_raw"
        ros_type = "sensor_msgs/msg/Image"
        image_resize = (16, 16)
        is_action = False

    streams, by_topic = _plan_streams(
        [Spec()],
        {"/camera/top/image_raw": "sensor_msgs/msg/Image"},
        {"observation.images.top"},
    )

    assert list(streams) == ["observation.images.top"]
    assert by_topic == {}


def test_rosbag_only_episode_keeps_existing_image_topic_path(tmp_path):
    class Spec:
        key = "observation.images.top"
        topic = "/camera/top/image_raw"
        ros_type = "sensor_msgs/msg/Image"
        image_resize = (16, 16)
        is_action = False

    assert discover_video_adapters(tmp_path) == {}
    streams, by_topic = _plan_streams([Spec()], {Spec.topic: Spec.ros_type})

    assert list(streams) == [Spec.key]
    assert by_topic == {Spec.topic: [Spec.key]}


def test_image_diagnostics_reports_annex_b_alignment_error(capsys):
    class Spec:
        image_resize = (16, 16)
        resample_policy = "hold"
        asof_tol_ms = 0

    stream = type("Stream", (), {"spec": Spec(), "ts": [1_000, 2_000_000]})()

    _log_image_stream_diagnostics(
        streams={"observation.images.top": stream},
        ticks_ns=np.asarray([1_000, 1_001_000], dtype=np.int64),
        step_ns=1_000_000,
        target_fps=1_000,
        video_sources={"observation.images.top": "Annex-B"},
    )

    output = capsys.readouterr().out
    assert "source=Annex-B" in output
    assert "max_alignment_error=1.000 ms" in output


def test_merge_integrity_report_preserves_episode_and_observation_context():
    info = {}

    _merge_integrity_report(info, 0, "observation.images.top", IntegrityReport(clean=True))
    _merge_integrity_report(
        info,
        1,
        "observation.images.wrist",
        IntegrityReport(
            clean=False,
            frame_gaps=[{"frame_index": 7, "lost_packets": 2, "reason": "rtp_sequence_gap"}],
        ),
    )

    assert info["integrity"] == {
        "clean": False,
        "frame_gaps": [
            {
                "episode_index": 1,
                "observation_key": "observation.images.wrist",
                "frame_index": 7,
                "lost_packets": 2,
                "reason": "rtp_sequence_gap",
            }
        ],
    }


def test_export_merges_annex_b_video_with_dds_action_and_state(tmp_path, monkeypatch):
    episode_dir = tmp_path / "episode_000001"
    episode_dir.mkdir()
    (episode_dir / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  storage_identifier: mcap\n  duration:\n    nanoseconds: 200000000\n",
        encoding="utf-8",
    )
    video_path = episode_dir / "observation.images.top.h264"
    codec = av.CodecContext.create("libx264", "w")
    codec.width = 16
    codec.height = 16
    codec.pix_fmt = "yuv420p"
    codec.options = {"profile": "baseline", "tune": "zerolatency", "x264-params": "bframes=0"}
    codec.open()
    packets = []
    for index in range(3):
        frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index * 40, dtype=np.uint8), format="rgb24")
        packets.extend(bytes(packet) for packet in codec.encode(frame))
    packets.extend(bytes(packet) for packet in codec.encode(None))
    video_path.write_bytes(b"".join(packets))
    sidecar_entries = [
        {
            "frame_index": index,
            "capture_timestamp_ns": 1_000_000_000 + index * 100_000_000,
            "rtp_timestamp": 90_000 + index * 9_000,
            "keyframe": index == 0,
            "lost_packets": 0,
            "session_generation": 1,
            "dropped": None,
        }
        for index in range(3)
    ]
    video_path.with_suffix(".h264.json").write_text(
        "".join(f"{json.dumps(entry)}\n" for entry in sidecar_entries), encoding="utf-8"
    )

    @dataclass
    class TopicType:
        name: str
        type: str

    class FakeReader:
        def __init__(self):
            self.messages = [
                ("/state", b"state0", 1_000_000_000),
                ("/action", b"action0", 1_000_000_000),
                ("/state", b"state1", 1_100_000_000),
                ("/action", b"action1", 1_100_000_000),
                ("/state", b"state2", 1_200_000_000),
                ("/action", b"action2", 1_200_000_000),
            ]

        def open(self, *_args):
            pass

        def get_all_topics_and_types(self):
            return [TopicType("/state", "test/State"), TopicType("/action", "test/Action")]

        def has_next(self):
            return bool(self.messages)

        def read_next(self):
            return self.messages.pop(0)

    class FakeMeta:
        def __init__(self):
            self.info = {"total_episodes": 0}

        def update_chunk_settings(self, **_kwargs):
            pass

    class FakeDataset:
        last = None

        def __init__(self, root):
            self.root = root
            self.meta = FakeMeta()
            self.frames = []
            self.saved = 0
            FakeDataset.last = self

        @classmethod
        def create(cls, *, root, **_kwargs):
            return cls(root)

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved += 1
            self.meta.info["total_episodes"] += 1

    image_spec = SimpleNamespace(
        key="observation.images.top",
        topic="/camera/top",
        ros_type="sensor_msgs/msg/Image",
        is_action=False,
        names=[],
        image_resize=(16, 16),
        resample_policy="hold",
        asof_tol_ms=0,
        stamp_src="header",
    )
    state_spec = SimpleNamespace(
        key="observation.state",
        topic="/state",
        ros_type="test/State",
        is_action=False,
        names=["joint"],
        image_resize=None,
        resample_policy="hold",
        asof_tol_ms=0,
        stamp_src="header",
    )
    action_spec = SimpleNamespace(
        key="action",
        topic="/action",
        ros_type="test/Action",
        is_action=True,
        names=["joint"],
        image_resize=None,
        resample_policy="hold",
        asof_tol_ms=0,
        stamp_src="header",
    )
    contract = SimpleNamespace(
        rate_hz=10,
        robot_type="test",
        observations=[],
        actions=[],
    )

    monkeypatch.setattr("dataset_tools.bag_to_lerobot._load_contract_from_robot_config", lambda _path: contract)
    monkeypatch.setattr("dataset_tools.bag_to_lerobot._resolve_fallback_conversion_config", lambda _path: {})
    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot.iter_specs", lambda _contract: [image_spec, state_spec, action_spec]
    )
    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot.feature_from_spec",
        lambda spec, _videos: (
            spec.key,
            {"dtype": "image", "shape": (16, 16, 3)}
            if spec.image_resize
            else {"dtype": "float32", "shape": (1,), "names": ["joint"]},
            spec.image_resize is not None,
        ),
    )
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.make_zero_pad", lambda feature: np.zeros(feature["shape"]))
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.rosbag2_py.SequentialReader", FakeReader)
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.deserialize_message", lambda data, _type: data)
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.get_message", lambda ros_type: ros_type)
    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot.decode_value",
        lambda _ros_type, data, _spec: np.asarray([0.1 if b"state" in data else 0.2], dtype=np.float32),
    )
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.LeRobotDataset", FakeDataset)
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.contract_fingerprint", lambda _contract: "fingerprint")

    export_bags_to_lerobot(
        [episode_dir],
        tmp_path / "robot.yaml",
        out_root=tmp_path / "output",
        use_videos=False,
    )

    dataset = FakeDataset.last
    assert dataset.saved == 1
    assert len(dataset.frames) == 3
    assert dataset.frames[0]["observation.images.top"].shape == (16, 16, 3)
    np.testing.assert_allclose(dataset.frames[0]["observation.state"], [0.1])
    np.testing.assert_allclose(dataset.frames[0]["action"], [0.2])
    assert dataset.meta.info["integrity"] == {"clean": True}


def test_clean_float_array_replaces_non_finite_values():
    arr = _clean_float_array([1.0, np.nan, np.inf, -np.inf], np.float32)

    assert arr.dtype == np.float32
    assert arr.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_clean_float_array_warns_for_non_current_features(caplog):
    with caplog.at_level("WARNING"):
        _clean_float_array([1.0, np.nan, 2.0], np.float32, feature_name="observation.state")
    assert any("observation.state" in rec.message for rec in caplog.records if rec.levelno >= 30)


def test_clean_float_array_silent_for_current(caplog):
    with caplog.at_level("WARNING"):
        _clean_float_array([1.0, np.nan, 2.0], np.float32, feature_name="observation.current")
    assert not any("non-finite" in rec.message for rec in caplog.records if rec.levelno >= 30)


def test_fallback_conversion_table_prefers_explicit_source_specs_over_legacy_pathsep(tmp_path):
    front = tmp_path / "front.json"
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    legacy = tmp_path / "legacy.json"
    for index, path in enumerate((front, left, right, legacy), start=1):
        path.write_text(json.dumps({"1": {"range_min": 1000 + index, "range_max": 3000 + index}}))

    specs = resolve_calibration_source_specs_from_config(
        {
            "ros2_control": {
                "xacro_args": {
                    "calib_file_front": str(front),
                    "calib_file_left": str(left),
                    "calib_file_right": str(right),
                }
            }
        }
    )

    table = _build_feature_conversion_table(
        feature_names=["joint1_front", "joint1_left", "joint1_right"],
        conversion_meta={},
        fallback_config={
            "norm_mode": "range_m100_100",
            "gripper_joints": [],
            "calibration_source_specs": specs,
            "calibration_file": str(legacy),
        },
    )

    assert len(table) == 3
    front_rad_min, front_rad_max, *_ = table[0]
    left_rad_min, left_rad_max, *_ = table[1]
    right_rad_min, right_rad_max, *_ = table[2]

    ticks_per_rad = 4096.0 / (2.0 * np.pi)
    assert front_rad_min == (1001 - 2048.0) / ticks_per_rad
    assert front_rad_max == (3001 - 2048.0) / ticks_per_rad
    assert left_rad_min == (1002 - 2048.0) / ticks_per_rad
    assert left_rad_max == (3002 - 2048.0) / ticks_per_rad
    assert right_rad_min == (1003 - 2048.0) / ticks_per_rad
    assert right_rad_max == (3003 - 2048.0) / ticks_per_rad
