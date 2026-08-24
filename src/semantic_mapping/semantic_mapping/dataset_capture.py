"""Finalize a supervised RGB-D LiDAR capture session into one compressed dataset archive."""

import argparse
import gzip
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

import yaml

from .offline_bag import OfflineTopicContract
from .offline_smoke import smoke_dataset

SEMANTIC_RECORDED_TOPIC_TYPES = {
    "/livox/lidar": "livox_ros_driver2/msg/CustomMsg",
    "/livox/imu": "sensor_msgs/msg/Imu",
    "/fast_lio/odometry_raw": "nav_msgs/msg/Odometry",
    "/odometry/filtered": "nav_msgs/msg/Odometry",
    "/cloud_registered_body": "sensor_msgs/msg/PointCloud2",
    "/scan": "sensor_msgs/msg/LaserScan",
    "/map": "nav_msgs/msg/OccupancyGrid",
    "/wheel/odom": "nav_msgs/msg/Odometry",
    "/cmd_vel": "geometry_msgs/msg/Twist",
    "/camera/front/image_raw": "sensor_msgs/msg/Image",
    "/camera/front/depth/image_rect_raw": "sensor_msgs/msg/Image",
    "/camera/front/aligned_depth_to_color/image_raw": "sensor_msgs/msg/Image",
    "/camera/front/camera_info": "sensor_msgs/msg/CameraInfo",
    "/camera/front/depth/camera_info": "sensor_msgs/msg/CameraInfo",
    "/camera/front/aligned_depth_to_color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
}
SEMANTIC_OPTIONAL_TOPICS = {"/wheel/odom", "/cmd_vel"}
SEMANTIC_REQUIRED_NONZERO_TOPICS = set(SEMANTIC_RECORDED_TOPIC_TYPES) - SEMANTIC_OPTIONAL_TOPICS


class BagReader(Protocol):
    def messages(self, topics: set[str]) -> Iterable[tuple[str, object, int]]: ...


class CompletedCommand(Protocol):
    returncode: int


class MessageHeader(Protocol):
    frame_id: str


class CameraInfoMessage(Protocol):
    header: MessageHeader
    width: int
    height: int
    distortion_model: str
    d: Sequence[float]
    k: Sequence[float]
    r: Sequence[float]
    p: Sequence[float]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid semantic dataset session state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid semantic dataset session state {path}: expected an object")
    return value


def validate_mcap_metadata(bag_dir: Path) -> dict[str, Any]:
    """Validate reindexed split files, actual topic records, and available file coverage."""
    metadata_path = bag_dir / "metadata.yaml"
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        information = metadata["rosbag2_bagfile_information"]
        relative_paths = information["relative_file_paths"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid rosbag metadata: {exc}") from exc
    if not isinstance(relative_paths, list) or not all(isinstance(item, str) for item in relative_paths):
        raise ValueError("rosbag metadata relative_file_paths must be a list of strings")
    actual = sorted(path.name for path in bag_dir.glob("*.mcap") if path.is_file())
    if not actual or sorted(relative_paths) != actual or len(relative_paths) != len(set(relative_paths)):
        raise ValueError(f"MCAP split list does not match bag directory: metadata={relative_paths}, actual={actual}")
    actual_topics = []
    for entry in information.get("topics_with_message_count", []):
        topic_metadata = entry.get("topic_metadata", {}) if isinstance(entry, dict) else {}
        name = topic_metadata.get("name")
        message_type = topic_metadata.get("type")
        count = entry.get("message_count")
        if isinstance(name, str) and isinstance(message_type, str) and isinstance(count, int):
            actual_topics.append({"name": name, "type": message_type, "message_count": count})
    actual_by_name = {}
    duplicate_topics = []
    for item in actual_topics:
        if item["name"] in actual_by_name:
            duplicate_topics.append(item["name"])
        actual_by_name[item["name"]] = item
    missing = sorted(set(SEMANTIC_REQUIRED_NONZERO_TOPICS) - set(actual_by_name))
    wrong_types = sorted(
        name
        for name, expected in SEMANTIC_RECORDED_TOPIC_TYPES.items()
        if name in actual_by_name and actual_by_name[name]["type"] != expected
    )
    empty = sorted(
        name
        for name in SEMANTIC_REQUIRED_NONZERO_TOPICS
        if name not in actual_by_name or actual_by_name[name]["message_count"] <= 0
    )
    if missing or wrong_types or empty or duplicate_topics:
        raise ValueError(
            f"required recorded topics are invalid: missing={missing}, wrong_types={wrong_types}, "
            f"empty={empty}, duplicates={sorted(set(duplicate_topics))}"
        )
    reported_optional = {
        name: actual_by_name[name]["message_count"]
        for name in sorted(SEMANTIC_OPTIONAL_TOPICS)
        if name in actual_by_name
    }
    missing_optional = sorted(SEMANTIC_OPTIONAL_TOPICS - set(actual_by_name))
    topic_validation = {
        "required_nonzero": sorted(SEMANTIC_REQUIRED_NONZERO_TOPICS),
        "reported_optional": reported_optional,
        "missing_optional": missing_optional,
    }
    time_coverage = _validate_time_coverage(information)
    return {
        "relative_file_paths": relative_paths,
        "metadata": metadata,
        "topics": sorted(actual_topics, key=lambda item: item["name"]),
        "topic_validation": topic_validation,
        "time_coverage": time_coverage,
    }


def _nanoseconds(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("nanoseconds_since_epoch", "nanoseconds"):
            if key in value:
                return int(value[key])
    if isinstance(value, int):
        return value
    return None


def _validate_time_coverage(information: dict[str, Any]) -> dict[str, Any]:
    files = information.get("files")
    if not isinstance(files, list) or not files:
        return {"status": "unavailable", "reason": "per-file time data absent"}
    coverage_fields = ("path", "starting_time", "duration")
    if not any(isinstance(item, dict) and any(field in item for field in coverage_fields) for item in files):
        return {"status": "unavailable", "reason": "per-file time data absent"}
    ranges = []
    covered_paths = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("metadata split file coverage is partial")
        path = item.get("path")
        start = _nanoseconds(item.get("starting_time"))
        duration = _nanoseconds(item.get("duration"))
        if not isinstance(path, str) or not path.strip() or start is None or duration is None:
            raise ValueError("metadata split file coverage is partial")
        covered_paths.append(Path(os.path.normpath(path)).as_posix())
        ranges.append((start, start + duration))
    relative_paths = [Path(os.path.normpath(path)).as_posix() for path in information["relative_file_paths"]]
    if len(covered_paths) != len(set(covered_paths)) or set(covered_paths) != set(relative_paths):
        raise ValueError(
            "metadata split file coverage paths must uniquely match relative_file_paths: "
            f"files={covered_paths}, relative_file_paths={relative_paths}"
        )
    top_start = _nanoseconds(information.get("starting_time"))
    top_duration = _nanoseconds(information.get("duration"))
    if top_start is None or top_duration is None:
        raise ValueError("metadata split times exist but top-level starting_time/duration is absent")
    files_start = min(start for start, _end in ranges)
    files_end = max(end for _start, end in ranges)
    top_end = top_start + top_duration
    if top_start > files_start or top_end < files_end:
        raise ValueError(
            f"metadata top-level time range does not cover split file time range: "
            f"top=[{top_start}, {top_end}], files=[{files_start}, {files_end}]"
        )
    return {
        "status": "validated",
        "top_start_ns": top_start,
        "top_end_ns": top_end,
        "files_start_ns": files_start,
        "files_end_ns": files_end,
    }


def _yaml_artifact(source: dict[str, Any], *, required: bool) -> dict[str, Any]:
    if source.get("status") == "missing":
        if required:
            raise ValueError("required pinned calibration source is missing")
        return {"status": "missing", "source": source.get("source", ""), "sha256": None, "data": None}
    path = Path(str(source["snapshot"]))
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid calibration source {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"invalid calibration source {path}: expected a mapping")
    document_status = str(document.get("status", "unknown"))
    status = str(source.get("status", document_status))
    if status != document_status:
        raise ValueError(f"pinned calibration source status changed: {path}")
    transform = document.get("transform", {})
    if "translation_m" in document:
        translation = [float(value) for value in document["translation_m"]]
        rpy_deg = [float(value) for value in document["rpy_deg"]]
        roll, pitch, yaw = [math.radians(value) for value in rpy_deg]
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        rotation = [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
        normalized = {
            "parent_frame": str(document.get("parent_frame", "base_link")),
            "child_frame": str(document.get("body_frame", "body")),
            "translation": translation,
            "rotation_xyzw": rotation,
        }
    else:
        normalized = {
            "parent_frame": str(transform.get("parent_frame", "")),
            "child_frame": str(transform.get("child_frame", "")),
            "translation": [float(value) for value in transform.get("translation", [])],
            "rotation_xyzw": [float(value) for value in transform.get("rotation_xyzw", [])],
        }
    if normalized["parent_frame"] != "base_link":
        raise ValueError(f"calibration source parent_frame must be base_link: {path}")
    expected_child = "body" if required else "camera_front_optical_frame"
    if normalized["child_frame"] != expected_child:
        raise ValueError(f"calibration source child_frame must be {expected_child}: {path}")
    if len(normalized["translation"]) != 3 or len(normalized["rotation_xyzw"]) != 4:
        raise ValueError(f"calibration source must contain translation[3] and rotation_xyzw[4]: {path}")
    actual_hash = _sha256(path)
    if actual_hash != source.get("sha256"):
        raise ValueError(f"pinned calibration source hash changed: {path}")
    return {
        "status": status,
        "source": source.get("source", ""),
        "sha256": actual_hash,
        "data": normalized,
    }


def _camera_info_artifact(message: CameraInfoMessage, topic: str) -> dict[str, Any]:
    data = {
        "frame_id": str(message.header.frame_id),
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": str(message.distortion_model),
        "D": [float(value) for value in message.d],
        "K": [float(value) for value in message.k],
        "R": [float(value) for value in message.r],
        "P": [float(value) for value in message.p],
    }
    if not data["frame_id"] or data["width"] <= 0 or data["height"] <= 0:
        raise ValueError("recorded CameraInfo has an invalid frame or dimensions")
    if len(data["K"]) != 9 or len(data["R"]) != 9 or len(data["P"]) != 12:
        raise ValueError("recorded CameraInfo has invalid K/R/P lengths")
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {"status": "recorded", "source": topic, "sha256": digest, "data": data}


def build_calibration_snapshot(
    bag_dir: Path,
    *,
    calibration_sources: dict[str, dict[str, Any]],
    camera_info_topic: str,
    reader_factory: Callable[[str], BagReader] | None = None,
) -> dict[str, Any]:
    """Snapshot current camera provenance, actual mount, and recorded CameraInfo."""
    camera_info = None
    if reader_factory is None:
        from .offline_bag import RosbagReader

        reader_factory = RosbagReader
    reader = reader_factory(str(bag_dir))
    for _topic, message, _stamp in reader.messages({camera_info_topic}):
        camera_info = message
        break
    if camera_info is None:
        raise ValueError(f"recorded bag has no CameraInfo messages on {camera_info_topic}")
    camera_artifact = _yaml_artifact(calibration_sources["base_to_front_camera"], required=False)
    return {
        "schema_version": "1.0",
        "status": "complete" if camera_artifact["status"] == "approved" else "calibration_incomplete",
        "generated_at": _timestamp(),
        "artifacts": {
            "base_to_front_camera": camera_artifact,
            "base_to_mid360": _yaml_artifact(calibration_sources["base_to_mid360"], required=True),
            "front_camera_intrinsics": _camera_info_artifact(cast(CameraInfoMessage, camera_info), camera_info_topic),
        },
    }


def _map_members(root: Path) -> list[Path]:
    yaml_files = sorted(path for path in (root / "map").glob("*.yaml") if path.is_file())
    pgm_files = sorted(path for path in (root / "map").glob("*.pgm") if path.is_file())
    if not yaml_files or not pgm_files:
        raise ValueError("dataset map directory must contain at least one YAML/PGM pair")
    return [*yaml_files, *pgm_files]


def _mcap_members(root: Path) -> list[Path]:
    files = sorted(path for path in (root / "bag").glob("*.mcap") if path.is_file())
    if not files:
        raise ValueError("dataset bag directory contains no MCAP files")
    return files


def _dataset_members(root: Path) -> list[Path]:
    members = [
        *_mcap_members(root),
        root / "bag" / "metadata.yaml",
        root / "bag" / "calibration_snapshot.json",
        *_map_members(root),
        root / "manifest.json",
        root / "SHA256SUMS",
        root / "README.md",
    ]
    missing = [path for path in members if not path.is_file()]
    if missing:
        raise ValueError(f"dataset is missing required artifact: {missing[0]}")
    return sorted(set(members), key=lambda path: path.relative_to(root).as_posix())


def write_checksums(root: Path) -> Path:
    """Write SHA-256 entries for every packaged file except the checksum file itself."""
    candidates = [path for path in _dataset_members_without_checksums(root) if path.name != "SHA256SUMS"]
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in candidates]
    output = root / "SHA256SUMS"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def verify_checksums(root: Path) -> None:
    """Verify every checksum generated for the packaged dataset."""
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or _sha256(root / relative) != expected:
            raise ValueError(f"dataset checksum verification failed: {relative or line}")


def _dataset_members_without_checksums(root: Path) -> list[Path]:
    members = [
        *_mcap_members(root),
        root / "bag" / "metadata.yaml",
        root / "bag" / "calibration_snapshot.json",
        *_map_members(root),
        root / "manifest.json",
        root / "README.md",
    ]
    missing = [path for path in members if not path.is_file()]
    if missing:
        raise ValueError(f"dataset is missing required artifact: {missing[0]}")
    return sorted(set(members), key=lambda path: path.relative_to(root).as_posix())


def create_deterministic_tar(root: Path) -> Path:
    """Create a gzip-compressed sibling tar with stable ordering and metadata."""
    archive_path = root.with_suffix(".tar.gz")
    temporary_path = archive_path.with_suffix(".gz.tmp")
    with (
        temporary_path.open("wb") as output,
        gzip.GzipFile(filename="", mode="wb", compresslevel=1, fileobj=output, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive,
    ):
        for path in _dataset_members(root):
            relative = path.relative_to(root).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    os.replace(temporary_path, archive_path)
    return archive_path


def _stop_recorder(state_path: Path, state: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    status = state.get("status")
    if status == "recorded":
        return state
    if status not in {"recording", "stopping"}:
        raise ValueError(f"semantic dataset recorder is not active: status={status}")
    if status == "recording":
        pid = int(state.get("supervisor_pid", 0))
        if pid <= 0:
            raise ValueError("semantic dataset session has no valid supervisor_pid")
        os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        state = _read_json(state_path)
        if state.get("status") == "recorded":
            return state
        if state.get("status") == "failed":
            raise ValueError(f"semantic dataset recorder failed: {state.get('error', 'unknown error')}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out after {timeout_sec:g}s waiting for semantic dataset recorder")


def _run_step(command: list[str], command_runner: Callable[..., CompletedCommand]) -> None:
    completed = command_runner(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with return code {completed.returncode}: {' '.join(command)}")


def promote_navigation_map(map_prefix: Path) -> Path:
    """Make the validated session map the default map consumed by Nav2."""
    from robot_navigation.save_lidar_map import promote_saved_map

    target_prefix = Path.home() / ".ros" / "ibrobot" / "maps" / "map"
    promote_saved_map(map_prefix, target_prefix)
    return target_prefix


def _record_save_failure(state_file: Path, status: str, stage: str, error: Exception) -> None:
    """Keep the handoff diagnosable when a finalized save step fails."""
    state_file = state_file.expanduser()
    try:
        state = _read_json(state_file)
        state["status"] = status
        state["failure_stage"] = stage
        state["failure_error"] = str(error)
        state["failure_at"] = _timestamp()
        if stage == "navigation_map_promotion":
            state["promotion_error"] = str(error)
            state["promotion_failed_at"] = state["failure_at"]
        state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError):
        return


def _refresh_navigation_map_archive(
    session_root: Path,
    status: str,
    *,
    source_prefix: Path,
    target_prefix: Path | None = None,
    error: Exception | None = None,
) -> Path:
    manifest_path = session_root / "manifest.json"
    manifest = _read_json(manifest_path)
    navigation_map = manifest.setdefault("navigation_map", {})
    navigation_map["status"] = status
    if error is not None:
        navigation_map["error"] = str(error)
    source_files = [source_prefix.with_suffix(".yaml"), source_prefix.with_suffix(".pgm")]
    navigation_map["source_sha256"] = {_path.name: _sha256(_path) for _path in source_files}
    if target_prefix is not None:
        target_files = [target_prefix.with_suffix(".yaml"), target_prefix.with_suffix(".pgm")]
        navigation_map["target_sha256"] = {_path.name: _sha256(_path) for _path in target_files}
        navigation_map["source_and_target_match"] = all(
            _sha256(source) == _sha256(target) for source, target in zip(source_files, target_files, strict=True)
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme_path = session_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    status_line = f"Navigation map promotion status: `{status}`\n"
    marker = "Navigation map promotion status: `"
    lines = readme.splitlines(keepends=True)
    if any(line.startswith(marker) for line in lines):
        lines = [status_line if line.startswith(marker) else line for line in lines]
    else:
        lines.append("\n" + status_line)
    readme_path.write_text("".join(lines), encoding="utf-8")
    write_checksums(session_root)
    return create_deterministic_tar(session_root)


def finalize_dataset(
    state_file: Path,
    *,
    map_prefix: Path | None = None,
    stop_timeout_sec: float = 30.0,
    command_runner: Callable[..., CompletedCommand] = subprocess.run,
    reader_factory: Callable[[str], BagReader] | None = None,
) -> Path:
    """Save the map, close/reindex the bag, then generate and package artifacts."""
    state_file = state_file.expanduser()
    state = _read_json(state_file)
    session_root = Path(str(state["session_root"]))
    bag_dir = session_root / "bag"
    output_prefix = map_prefix.expanduser() if map_prefix else session_root / "map" / "map"
    try:
        output_prefix.resolve().relative_to(session_root.resolve())
    except ValueError as exc:
        raise ValueError("map prefix must remain inside the semantic dataset session") from exc
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    state = _stop_recorder(state_file, state, stop_timeout_sec)
    _run_step(
        ["ros2", "run", "robot_navigation", "save_lidar_map", "-f", str(output_prefix), "-t", "/map"],
        command_runner,
    )
    for map_path in (output_prefix.with_suffix(".yaml"), output_prefix.with_suffix(".pgm")):
        if not map_path.is_file() or map_path.stat().st_size == 0:
            raise ValueError(f"map save did not produce a usable artifact: {map_path}")
    _run_step(["ros2", "bag", "reindex", "--storage", "mcap", str(bag_dir)], command_runner)
    metadata = validate_mcap_metadata(bag_dir)

    snapshot = build_calibration_snapshot(
        bag_dir,
        calibration_sources=state["calibration_sources"],
        camera_info_topic=str(state["camera_info_topic"]),
        reader_factory=reader_factory,
    )
    snapshot_path = bag_dir / "calibration_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    map_files = [
        output_prefix.with_suffix(".yaml").relative_to(session_root).as_posix(),
        output_prefix.with_suffix(".pgm").relative_to(session_root).as_posix(),
    ]
    manifest = {
        "schema_version": "1.0",
        "status": snapshot["status"],
        "dataset_id": state["session_id"],
        "profile": state["profile"],
        "generated_at": _timestamp(),
        "topics": metadata["topics"],
        "bag": {
            "storage": "mcap",
            "relative_file_paths": metadata["relative_file_paths"],
            "time_coverage": metadata["time_coverage"],
            "topic_validation": metadata["topic_validation"],
        },
        "map_files": map_files,
        "navigation_map": {
            "status": "pending_validation",
            "target_files": [
                "~/.ros/ibrobot/maps/map.yaml",
                "~/.ros/ibrobot/maps/map.pgm",
            ],
            "source_files": map_files,
        },
        "calibration_snapshot": "bag/calibration_snapshot.json",
        "camera_calibration_status": snapshot["artifacts"]["base_to_front_camera"]["status"],
    }
    (session_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (session_root / "README.md").write_text(
        "# RGB-D LiDAR Mapping Dataset\n\n"
        f"Profile: `{state['profile']}`\n\n"
        f"Dataset status: `{snapshot['status']}`\n\n"
        "The `bag/` directory contains the reindexed MCAP recording and calibration snapshot. "
        "The `map/` directory contains the slam_toolbox map saved from the same session. "
        "After checksum and offline geometry validation, this exact YAML/PGM pair is promoted to "
        "`~/.ros/ibrobot/maps/map.yaml` and `map.pgm` for Nav2.\n"
        "Navigation map promotion status: `pending_validation`\n",
        encoding="utf-8",
    )
    write_checksums(session_root)
    print(f"Compressing semantic dataset to {session_root.with_suffix('.tar.gz')} (gzip level 1)...", flush=True)
    archive_path = create_deterministic_tar(session_root)
    state["status"] = "finalized" if snapshot["status"] == "complete" else "calibration_incomplete"
    state["finalized_at"] = _timestamp()
    state["archive_path"] = str(archive_path)
    state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return archive_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save and validate the current semantic mapping dataset.")
    parser.add_argument(
        "--session-file",
        default="~/.ros/ibrobot/semantic_mapping/current.json",
        help="Recorder session handoff JSON",
    )
    parser.add_argument("--map-prefix", help="Optional map output prefix inside the session map directory")
    parser.add_argument("--stop-timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    map_prefix = Path(args.map_prefix).expanduser() if args.map_prefix else None
    state_file = Path(args.session_file)
    try:
        try:
            archive = finalize_dataset(state_file, map_prefix=map_prefix, stop_timeout_sec=args.stop_timeout)
        except (KeyError, OSError, RuntimeError, TimeoutError, ValueError, yaml.YAMLError) as exc:
            _record_save_failure(state_file, "semantic_map_finalization_failed", "finalization", exc)
            raise
        session_root = archive.with_suffix("").with_suffix("")
        try:
            verify_checksums(session_root)
        except (OSError, ValueError) as exc:
            _record_save_failure(state_file, "semantic_map_checksum_failed", "checksum_validation", exc)
            raise
        source_prefix = map_prefix or session_root / "map" / "map"
        try:
            smoke = smoke_dataset(
                session_root,
                topics=OfflineTopicContract(
                    "/camera/front/image_raw",
                    "/camera/front/aligned_depth_to_color/image_raw",
                    "/camera/front/camera_info",
                ),
            )
            if not smoke["geometry_ready"]:
                raise ValueError(f"offline validation failed: calibration_status={smoke['calibration_status']}")
        except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
            _record_save_failure(state_file, "semantic_map_offline_validation_failed", "offline_validation", exc)
            with suppress(OSError, KeyError, ValueError):
                archive = _refresh_navigation_map_archive(
                    session_root,
                    "offline_validation_failed",
                    source_prefix=source_prefix,
                    error=exc,
                )
            raise
        try:
            target_prefix = promote_navigation_map(source_prefix)
        except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
            _record_save_failure(state_file, "navigation_map_promotion_failed", "navigation_map_promotion", exc)
            with suppress(OSError, KeyError, ValueError):
                archive = _refresh_navigation_map_archive(
                    session_root,
                    "promotion_failed",
                    source_prefix=source_prefix,
                    error=exc,
                )
            raise
        try:
            archive = _refresh_navigation_map_archive(
                session_root,
                "promoted",
                source_prefix=source_prefix,
                target_prefix=target_prefix,
            )
            verify_checksums(session_root)
        except (KeyError, OSError, ValueError) as exc:
            _record_save_failure(state_file, "navigation_map_audit_failed", "archive_refresh", exc)
            raise
    except (KeyError, OSError, RuntimeError, TimeoutError, ValueError, yaml.YAMLError) as exc:
        print(f"Semantic map save failed: {exc}", file=sys.stderr)
        return 1
    archive_size_gib = archive.stat().st_size / (1024**3)
    print(f"Semantic map saved: session={session_root} archive={archive} archive_size={archive_size_gib:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
