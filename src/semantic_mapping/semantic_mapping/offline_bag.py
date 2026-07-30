"""Validated two-pass rosbag source for offline semantic mapping."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer, TransformException

from .database import SemanticMapManifest


@dataclass(frozen=True)
class OfflineTopicContract:
    rgb: str
    aligned_depth: str
    camera_info: str
    dynamic_tf: str = "/tf"
    static_tf: str = "/tf_static"

    @property
    def required(self) -> set[str]:
        return {self.rgb, self.aligned_depth, self.camera_info, self.dynamic_tf, self.static_tf}


@dataclass(frozen=True)
class OfflineFrameMessages:
    rgb: object
    depth: object
    camera_info: object
    transform: object
    stamp_ns: int


@dataclass
class OfflineBagDiagnostics:
    topic_messages: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    synchronized_frames: int = 0
    rejected_sync: int = 0
    rejected_tf: int = 0
    deserialization_failures: int = 0


def message_stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def create_run_manifest(
    *,
    global_frame: str,
    geometry_map_id: str,
    geometry_map_hash: str,
    localization_session_id: str,
    calibration_id: str,
    urdf_hash: str,
    coordinate_convention: str,
    semantic_identities: dict,
    bag_path: str,
    topics: OfflineTopicContract,
) -> SemanticMapManifest:
    manifest = SemanticMapManifest(
        global_frame=global_frame,
        geometry_map_id=geometry_map_id,
        geometry_map_hash=geometry_map_hash,
        localization_session_id=localization_session_id,
        calibration_id=calibration_id,
        urdf_hash=urdf_hash,
        coordinate_convention=coordinate_convention,
        semantic_identities=semantic_identities,
        settings={
            "source": "rosbag2",
            "bag_path": str(Path(bag_path).expanduser().resolve()),
            "topics": {
                "rgb": topics.rgb,
                "aligned_depth": topics.aligned_depth,
                "camera_info": topics.camera_info,
                "tf": topics.dynamic_tf,
                "tf_static": topics.static_tf,
            },
        },
    )
    manifest.validate()
    return manifest


class RosbagReader:
    """Open a rosbag for each pass and deserialize selected topics."""

    def __init__(self, uri: str, storage_id: str = ""):
        self.uri = str(Path(uri).expanduser())
        self.storage_id = storage_id

    def _open(self):
        import rosbag2_py

        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=self.uri, storage_id=self.storage_id),
            rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
        return reader

    def topic_types(self) -> dict[str, str]:
        reader = self._open()
        return {item.name: item.type for item in reader.get_all_topics_and_types()}

    def duration_ns(self) -> int:
        metadata = self._open().get_metadata()
        return int(metadata.duration.nanoseconds)

    def messages(self, topics: set[str]):
        import rosbag2_py

        reader = self._open()
        types = {item.name: item.type for item in reader.get_all_topics_and_types()}
        reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(topics)))
        message_types = {topic: get_message(types[topic]) for topic in topics if topic in types}
        while reader.has_next():
            topic, data, bag_stamp_ns = reader.read_next()
            yield topic, deserialize_message(data, message_types[topic]), int(bag_stamp_ns)


class OfflineBagSource:
    EXPECTED_TYPES = {
        "rgb": "sensor_msgs/msg/Image",
        "aligned_depth": "sensor_msgs/msg/Image",
        "camera_info": "sensor_msgs/msg/CameraInfo",
        "dynamic_tf": "tf2_msgs/msg/TFMessage",
        "static_tf": "tf2_msgs/msg/TFMessage",
    }

    def __init__(
        self,
        reader,
        topics: OfflineTopicContract,
        *,
        global_frame: str,
        sync_slop_ns: int,
    ):
        if sync_slop_ns < 0:
            raise ValueError("sync slop must be non-negative")
        self.reader = reader
        self.topics = topics
        self.global_frame = global_frame
        self.sync_slop_ns = sync_slop_ns
        self.diagnostics = OfflineBagDiagnostics()

    def validate_topics(self) -> None:
        types = self.reader.topic_types()
        missing = sorted(self.topics.required - set(types))
        if missing:
            raise ValueError(f"rosbag is missing required topics: {', '.join(missing)}")
        for field_name, expected in self.EXPECTED_TYPES.items():
            topic = getattr(self.topics, field_name)
            if types[topic] != expected:
                raise ValueError(f"topic {topic} must use {expected}, got {types[topic]}")

    def build_tf_buffer(self) -> Buffer:
        duration_ns = max(int(self.reader.duration_ns()), 1_000_000_000)
        buffer = Buffer(cache_time=Duration(nanoseconds=duration_ns + 1_000_000_000))
        for topic, message, _ in self.reader.messages({self.topics.dynamic_tf, self.topics.static_tf}):
            self.diagnostics.topic_messages[topic] += 1
            for transform in message.transforms:
                if topic == self.topics.static_tf:
                    buffer.set_transform_static(transform, "offline_rosbag")
                else:
                    buffer.set_transform(transform, "offline_rosbag")
        return buffer

    @staticmethod
    def _nearest(messages: list, stamp_ns: int, slop_ns: int, *, allow_zero: bool = False):
        candidates = []
        for message in messages:
            candidate_stamp = message_stamp_ns(message)
            if allow_zero and candidate_stamp == 0:
                candidates.append((0, message))
            else:
                candidates.append((abs(candidate_stamp - stamp_ns), message))
        if not candidates:
            return None
        difference, message = min(candidates, key=lambda item: item[0])
        return message if difference <= slop_ns else None

    def frames(self, tf_buffer: Buffer):
        streams = defaultdict(list)
        selected = {self.topics.rgb, self.topics.aligned_depth, self.topics.camera_info}
        for topic, message, _ in self.reader.messages(selected):
            self.diagnostics.topic_messages[topic] += 1
            streams[topic].append(message)
        if not streams[self.topics.rgb]:
            raise ValueError("rosbag RGB stream has no usable messages")
        if not streams[self.topics.aligned_depth]:
            raise ValueError("rosbag aligned-depth stream has no usable messages")
        if not streams[self.topics.camera_info]:
            raise ValueError("rosbag CameraInfo stream has no usable messages")

        for rgb in sorted(streams[self.topics.rgb], key=message_stamp_ns):
            stamp_ns = message_stamp_ns(rgb)
            depth = self._nearest(streams[self.topics.aligned_depth], stamp_ns, self.sync_slop_ns)
            camera_info = self._nearest(streams[self.topics.camera_info], stamp_ns, self.sync_slop_ns, allow_zero=True)
            if depth is None or camera_info is None:
                self.diagnostics.rejected_sync += 1
                continue
            camera_frame = rgb.header.frame_id or camera_info.header.frame_id
            try:
                transform = tf_buffer.lookup_transform(self.global_frame, camera_frame, Time(nanoseconds=stamp_ns))
            except TransformException:
                self.diagnostics.rejected_tf += 1
                continue
            self.diagnostics.synchronized_frames += 1
            yield OfflineFrameMessages(rgb, depth, camera_info, transform, stamp_ns)
