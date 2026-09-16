"""Validate rosbag2 MCAP inputs for offline FAST-Calib export."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsFloat

import yaml

REQUIRED_TOPIC_TYPES = {
    "/livox/lidar": "livox_ros_driver2/msg/CustomMsg",
    "/livox/imu": "sensor_msgs/msg/Imu",
    "/cloud_registered_body": "sensor_msgs/msg/PointCloud2",
    "/odometry/filtered": "nav_msgs/msg/Odometry",
    "/camera/front/image_raw": "sensor_msgs/msg/Image",
    "/camera/front/camera_info": "sensor_msgs/msg/CameraInfo",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
}


@dataclass(frozen=True)
class FastCalibBag:
    path: Path
    metadata: dict[str, Any]
    storage_files: tuple[Path, ...]
    duration_s: float
    topic_counts: dict[str, int]


def camera_coefficients(values: Iterable[SupportsFloat]) -> list[float]:
    """Normalize generated ROS numeric sequences for portable YAML output."""
    return [float(value) for value in values]


def validate_fast_calib_bag(path: str | Path) -> FastCalibBag:
    """Validate storage, a positive duration, and all required topics."""
    root = Path(path).expanduser()
    metadata_path = root / "metadata.yaml"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError(f"bag metadata is missing or not regular: {metadata_path}")
    try:
        metadata = yaml.safe_load(metadata_path.read_bytes())
        information = metadata["rosbag2_bagfile_information"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise ValueError(f"bag metadata is invalid: {exc}") from exc
    if information.get("storage_identifier") != "mcap":
        raise ValueError("bag storage_identifier must equal mcap")
    duration = information.get("duration", {}).get("nanoseconds")
    if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
        raise ValueError("bag duration is invalid")
    relative_paths = information.get("relative_file_paths")
    if not isinstance(relative_paths, list) or not relative_paths:
        raise ValueError("bag metadata must declare storage files")
    storage_files = []
    for value in relative_paths:
        relative = Path(value) if isinstance(value, str) else Path()
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != value:
            raise ValueError(f"invalid storage path: {value!r}")
        storage = root / relative
        if storage.is_symlink() or not storage.is_file():
            raise ValueError(f"missing or invalid storage file: {value}")
        storage_files.append(storage)
    observed = {}
    for entry in information.get("topics_with_message_count", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("topic_metadata"), dict):
            raise ValueError("bag topic entry is invalid")
        count = entry.get("message_count")
        topic = entry["topic_metadata"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("bag topic count is invalid")
        observed[topic.get("name")] = (topic.get("type"), count)
    counts = {}
    for name, expected_type in REQUIRED_TOPIC_TYPES.items():
        if name not in observed:
            raise ValueError(f"bag is missing required topic: {name}")
        actual_type, count = observed[name]
        if actual_type != expected_type:
            raise ValueError(f"topic {name} type is {actual_type}; expected {expected_type}")
        if count <= 0:
            raise ValueError(f"topic has no messages: {name}")
        counts[name] = count
    return FastCalibBag(root, metadata, tuple(storage_files), duration / 1_000_000_000, counts)
