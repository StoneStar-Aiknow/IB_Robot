"""Export complete static FAST-Calib scenes from rosbag2 MCAP."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml

from robot_calibration.bag import REQUIRED_TOPIC_TYPES, camera_coefficients, validate_fast_calib_bag
from robot_calibration.offline import REQUIRED_SCENES


def _transform(translation: list[float], quaternion: list[float]) -> np.ndarray:
    x, y, z, w = quaternion
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _finite_points(frames: list[np.ndarray]) -> np.ndarray:
    valid = [points[np.isfinite(points).all(axis=1)] for points in frames if points.ndim == 2 and points.shape[1] == 3]
    valid = [points for points in valid if len(points)]
    if not valid:
        raise ValueError("scene contains no finite points")
    return np.concatenate(valid)


def _resolve_body_from_livox(static_transforms: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    direct = static_transforms.get(("body", "livox_frame"))
    if direct is not None:
        return direct
    base_from_body = static_transforms.get(("base_link", "body"))
    base_from_livox = static_transforms.get(("base_link", "livox_frame"))
    if base_from_body is None or base_from_livox is None:
        raise ValueError("static transform body -> livox_frame is required")
    return np.linalg.inv(base_from_body) @ base_from_livox


def _write_pcd(path: Path, points: np.ndarray) -> None:
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n")
        stream.write("FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        stream.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        np.savetxt(stream, points, fmt="%.6f %.6f %.6f")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_scene(bag_path: Path, output: Path, counts: dict[str, int]) -> dict:
    import cv2
    from cv_bridge import CvBridge
    from rclpy.serialization import deserialize_message, serialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, SequentialWriter, StorageOptions, TopicMetadata
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header

    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag_path), storage_id="mcap"), ConverterOptions("cdr", "cdr"))
    type_map = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    if any(type_map.get(topic) != expected for topic, expected in REQUIRED_TOPIC_TYPES.items()):
        raise ValueError("reader topic types differ from validated metadata")
    decoded_types = {topic: get_message(type_map[topic]) for topic in REQUIRED_TOPIC_TYPES}
    image_target = counts["/camera/front/image_raw"] // 2
    image_seen = 0
    selected_image = None
    selected_timestamp = None
    camera_info = None
    image_frames: set[str] = set()
    info_frames: set[str] = set()
    body_frames: set[str] = set()
    raw_frames: set[str] = set()
    body_clouds = []
    raw_clouds = []
    static_transforms = {}
    bridge = CvBridge()
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic not in decoded_types:
            continue
        message = deserialize_message(data, decoded_types[topic])
        frame = str(getattr(getattr(message, "header", None), "frame_id", ""))
        if topic == "/camera/front/image_raw":
            image_frames.add(frame)
            if image_seen == image_target:
                selected_image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                selected_timestamp = int(timestamp)
            image_seen += 1
        elif topic == "/camera/front/camera_info":
            info_frames.add(frame)
            if camera_info is None:
                camera_info = {
                    "schema_version": "1.0",
                    "frame_id": frame,
                    "width": int(message.width),
                    "height": int(message.height),
                    "distortion_model": str(message.distortion_model),
                    "D": camera_coefficients(message.d),
                    "K": camera_coefficients(message.k),
                    "R": camera_coefficients(message.r),
                    "P": camera_coefficients(message.p),
                }
        elif topic == "/cloud_registered_body":
            body_frames.add(frame)
            body_clouds.append(point_cloud2.read_points_numpy(message, field_names=["x", "y", "z"]))
        elif topic == "/livox/lidar":
            raw_frames.add(frame)
            raw_clouds.append(np.array([(point.x, point.y, point.z) for point in message.points], dtype=np.float32))
        elif topic == "/tf_static":
            for item in message.transforms:
                edge = (item.header.frame_id, item.child_frame_id)
                if edge in {
                    ("body", "livox_frame"),
                    ("base_link", "body"),
                    ("base_link", "livox_frame"),
                }:
                    t, q = item.transform.translation, item.transform.rotation
                    matrix = _transform([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
                    previous = static_transforms.get(edge)
                    if previous is not None and not np.allclose(previous, matrix, atol=1e-9):
                        raise ValueError(f"conflicting static transform for {edge[0]} -> {edge[1]}")
                    static_transforms[edge] = matrix
    if selected_image is None or selected_timestamp is None or camera_info is None:
        raise ValueError("scene has no selected RGB or CameraInfo")
    if len(image_frames) != 1 or image_frames != info_frames:
        raise ValueError("RGB and CameraInfo frame_id must be stable and equal")
    if len(body_frames) != 1 or "" in body_frames or raw_frames != {"livox_frame"}:
        raise ValueError("point cloud frame contract is invalid")
    body_points = _finite_points(body_clouds)
    raw_points = _finite_points(raw_clouds)
    body_from_livox = _resolve_body_from_livox(static_transforms)
    dense_points = raw_points @ body_from_livox[:3, :3].T + body_from_livox[:3, 3]
    image_path = output / "image.png"
    if not cv2.imwrite(str(image_path), selected_image):
        raise OSError(f"unable to write {image_path}")
    camera_path = output / "camera_info.yaml"
    camera_path.write_text(yaml.safe_dump(camera_info, sort_keys=False), encoding="utf-8")
    paths = [
        output / name
        for name in ("cloud_body_accumulated.pcd", "cloud_livox_accumulated.pcd", "cloud_dense_body_from_livox.pcd")
    ]
    for path, points in zip(paths, (body_points, raw_points, dense_points), strict=True):
        _write_pcd(path, points)
    dense_bag = output / "dense_bag_v1"
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=str(dense_bag), storage_id="mcap"), ConverterOptions("", ""))
    writer.create_topic(
        TopicMetadata(name="/cloud_dense_body", type="sensor_msgs/msg/PointCloud2", serialization_format="cdr")
    )
    header = Header()
    header.frame_id = "body"
    header.stamp.sec, header.stamp.nanosec = divmod(selected_timestamp, 1_000_000_000)
    cloud: PointCloud2 = point_cloud2.create_cloud_xyz32(header, dense_points.astype(np.float32, copy=False))
    writer.write("/cloud_dense_body", serialize_message(cloud), selected_timestamp)
    files = [image_path, camera_path, *paths]
    return {
        "schema_version": "1.0",
        "selected_image_index": image_target,
        "selected_image_timestamp_ns": selected_timestamp,
        "point_counts": {
            "body_accumulated": len(body_points),
            "raw_livox_accumulated": len(raw_points),
            "dense_body_from_livox": len(dense_points),
        },
        "body_from_livox": body_from_livox.tolist(),
        "files": {path.name: {"sha256": _sha256(path), "size": path.stat().st_size} for path in files},
        "dense_bag": {"path": dense_bag.name, "topic": "/cloud_dense_body", "frame_id": "body"},
    }


def export_scene(scene: Path, output: Path) -> dict:
    """Decode one validated scene through private staging."""
    if os.path.lexists(output):
        raise FileExistsError(output)
    bag = validate_fast_calib_bag(scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        manifest = _decode_scene(bag.path, staging, bag.topic_counts)
        (staging / "scene_export.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True), encoding="ascii"
        )
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def export_capture(capture: Path, output: Path) -> dict[str, dict]:
    """Export exactly four required scenes and remove partial output on failure."""
    if os.path.lexists(output):
        raise FileExistsError(output)
    manifests = {}
    try:
        for scene in REQUIRED_SCENES:
            manifests[scene] = export_scene(capture / scene, output / scene)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return manifests
