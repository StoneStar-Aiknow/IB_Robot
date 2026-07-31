from types import SimpleNamespace

import pytest
from tf2_ros import TransformException

from semantic_mapping.offline_bag import OfflineBagSource, OfflineTopicContract, create_run_manifest


def _identities():
    common = {"preprocessing_contract": "pre-v1", "output_semantics": "output-v1"}
    return {
        "sam2": {"logical_model_revision": "sam2@v1", **common},
        "ram_plus": {"logical_model_revision": "ram-plus@v1", **common},
        "siglip2_image": {
            "logical_model_revision": "siglip2@v1",
            **common,
            "embedding": {
                "embedding_space_id": "siglip2-space:v1",
                "dimension": 2,
                "normalization": "l2",
                "image_preprocessing": "image-v1",
                "text_preprocessing": "text-v1",
            },
        },
    }


def _stamp(nanoseconds):
    return SimpleNamespace(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def _message(stamp_ns, frame="d435_color_optical_frame"):
    return SimpleNamespace(header=SimpleNamespace(stamp=_stamp(stamp_ns), frame_id=frame))


class _Reader:
    def __init__(self, types, messages):
        self._types = types
        self._messages = messages

    def topic_types(self):
        return self._types

    def messages(self, topics):
        for topic, message in self._messages:
            if topic in topics:
                yield topic, message, 0


class _Buffer:
    def __init__(self, missing_stamps=()):
        self.missing_stamps = set(missing_stamps)
        self.requests = []

    def lookup_transform(self, target, source, stamp):
        stamp_ns = stamp.nanoseconds
        self.requests.append((target, source, stamp_ns))
        if stamp_ns in self.missing_stamps:
            raise TransformException("missing historical transform")
        return SimpleNamespace()


@pytest.fixture
def topics():
    return OfflineTopicContract("/rgb", "/depth", "/info")


def _types(topics):
    return {
        topics.rgb: "sensor_msgs/msg/Image",
        topics.aligned_depth: "sensor_msgs/msg/Image",
        topics.camera_info: "sensor_msgs/msg/CameraInfo",
        topics.dynamic_tf: "tf2_msgs/msg/TFMessage",
        topics.static_tf: "tf2_msgs/msg/TFMessage",
    }


def test_required_topic_validation_fails_before_processing(topics):
    types = _types(topics)
    del types[topics.aligned_depth]
    source = OfflineBagSource(_Reader(types, []), topics, global_frame="map", sync_slop_ns=20_000_000)

    with pytest.raises(ValueError, match="/depth"):
        source.validate_topics()


def test_frames_use_rgb_timestamp_and_accept_static_camera_info(topics):
    messages = [
        (topics.rgb, _message(1_000_000_000)),
        (topics.aligned_depth, _message(1_010_000_000)),
        (topics.camera_info, _message(0)),
    ]
    source = OfflineBagSource(_Reader(_types(topics), messages), topics, global_frame="map", sync_slop_ns=20_000_000)
    buffer = _Buffer()

    frames = list(source.frames(buffer))

    assert len(frames) == 1
    assert frames[0].stamp_ns == 1_000_000_000
    assert buffer.requests == [("map", "d435_color_optical_frame", 1_000_000_000)]
    assert source.diagnostics.synchronized_frames == 1


def test_frame_without_historical_tf_is_counted_and_not_emitted(topics):
    messages = [
        (topics.rgb, _message(1_000_000_000)),
        (topics.aligned_depth, _message(1_000_000_000)),
        (topics.camera_info, _message(1_000_000_000)),
    ]
    source = OfflineBagSource(_Reader(_types(topics), messages), topics, global_frame="map", sync_slop_ns=1)

    assert list(source.frames(_Buffer(missing_stamps={1_000_000_000}))) == []
    assert source.diagnostics.rejected_tf == 1


def test_run_manifest_records_bag_topics_and_requires_map_identity(topics, tmp_path):
    manifest = create_run_manifest(
        global_frame="map",
        geometry_map_id="warehouse",
        geometry_map_hash="map-hash",
        localization_session_id="session",
        calibration_id="calibration",
        urdf_hash="urdf-hash",
        coordinate_convention="ros-rep-103-map-enu",
        semantic_identities=_identities(),
        bag_path=str(tmp_path / "bag"),
        topics=topics,
    )

    assert manifest.settings["source"] == "rosbag2"
    assert manifest.settings["topics"]["aligned_depth"] == "/depth"
    with pytest.raises(ValueError, match="geometry_map_hash"):
        create_run_manifest(
            global_frame="map",
            geometry_map_id="warehouse",
            geometry_map_hash="",
            localization_session_id="session",
            calibration_id="calibration",
            urdf_hash="urdf-hash",
            coordinate_convention="ros-rep-103-map-enu",
            semantic_identities=_identities(),
            bag_path=str(tmp_path / "bag"),
            topics=topics,
        )
