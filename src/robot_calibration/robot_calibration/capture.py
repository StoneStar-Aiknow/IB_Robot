"""Validate and seal a ROS-independent capture manifest summary."""

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SCENES = ("scene-01", "scene-02", "scene-03", "scene-04-test")
REQUIRED_TOPICS = (
    "/livox/lidar",
    "/livox/imu",
    "/cloud_registered_body",
    "/odometry/filtered",
    "/camera/front/image_raw",
    "/camera/front/camera_info",
    "/tf",
    "/tf_static",
)


class CaptureError(ValueError):
    """Raised when a capture cannot be sealed."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CaptureError("capture file path must be normalized and relative")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CaptureError("capture file path must be normalized and relative")
    return path


def _validate_scene(scene: dict[str, Any]) -> None:
    scene_id = scene.get("scene_id")
    if scene_id not in REQUIRED_SCENES:
        raise CaptureError(f"unknown scene {scene_id!r}")
    duration = scene.get("duration_s")
    if not isinstance(duration, int | float) or duration <= 0:
        raise CaptureError(f"scene {scene_id} duration must be positive")
    if scene.get("stationary_windows", 0) < 1:
        raise CaptureError(f"scene {scene_id} has no stationary window")
    topics = scene.get("topics")
    if not isinstance(topics, dict):
        raise CaptureError(f"scene {scene_id} topics are missing")
    for topic in REQUIRED_TOPICS:
        sample = topics.get(topic)
        if not isinstance(sample, dict) or sample.get("count", 0) <= 0:
            raise CaptureError(f"scene {scene_id} topic {topic} has no samples")
    if {"base_link -> body", "body -> camera_front_link"} - set(scene.get("tf_edges", [])):
        raise CaptureError(f"scene {scene_id} TF chain is incomplete")


def finalize_capture(
    root: Path,
    capture_id: str,
    scenes: list[dict[str, Any]],
    devices: dict[str, str],
    *,
    source: Path,
    legacy_transfer_manifest_sha256: str | None = None,
) -> Path:
    """Seal a new capture directory without overwriting an existing capture."""
    if not capture_id or Path(capture_id).name != capture_id or capture_id in {".", ".."}:
        raise CaptureError("capture_id must be a single safe path component")
    if len(scenes) != 4 or {scene.get("scene_id") for scene in scenes} != set(REQUIRED_SCENES):
        raise CaptureError("capture must contain exactly four required scenes")
    for scene in scenes:
        _validate_scene(scene)
    destination = root / capture_id
    if destination.exists():
        raise CaptureError(f"capture {capture_id} already exists")
    staging = root / f".{capture_id}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        files = []
        declared = set()
        for scene in scenes:
            entries = scene.get("files")
            if not isinstance(entries, list) or not entries:
                raise CaptureError(f"scene {scene['scene_id']} files are missing")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise CaptureError("capture file entry must be a mapping")
                relative = _relative_path(entry.get("path"))
                relative_text = relative.as_posix()
                if relative_text in declared:
                    raise CaptureError(f"duplicate capture file {relative_text}")
                declared.add(relative_text)
                source_file = source / relative
                if source_file.is_symlink() or not source_file.is_file():
                    raise CaptureError(f"capture file is missing or not regular: {relative_text}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, target)
                files.append({"path": relative_text, "size": target.stat().st_size, "sha256": _sha256(target)})
        manifest = {
            "schema_version": 1,
            "capture_id": capture_id,
            "devices": devices,
            "scenes": {scene["scene_id"]: scene for scene in scenes},
            "files": sorted(files, key=lambda item: item["path"]),
            "sealed": True,
            "sealed_at": datetime.now(timezone.utc).isoformat(),
        }
        if legacy_transfer_manifest_sha256 is not None:
            manifest["legacy_transfer_manifest_sha256"] = legacy_transfer_manifest_sha256
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_bytes(_canonical(manifest))
        (staging / "FINALIZED").write_text("\n", encoding="ascii")
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def import_legacy_capture(source: Path, root: Path, *, devices: dict[str, str]) -> Path:
    """Validate and seal a read-only historical four-scene transfer bundle."""
    transfer_path = source / "transfer_manifest.json"
    try:
        content = transfer_path.read_bytes()
        transfer = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"legacy transfer manifest is invalid: {exc}") from exc
    if not isinstance(transfer, dict) or set(transfer) != {"schema_version", "bundle_id", "bundle_type", "files"}:
        raise CaptureError("legacy transfer manifest shape is invalid")
    if transfer["schema_version"] != "1.0" or transfer["bundle_type"] != "fast_calib_capture":
        raise CaptureError("legacy transfer manifest contract is invalid")
    entries = transfer["files"]
    if not isinstance(entries, list) or not entries:
        raise CaptureError("legacy transfer manifest files are missing")
    declared: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise CaptureError("legacy transfer file entry is invalid")
        relative = _relative_path(entry["path"])
        relative_text = relative.as_posix()
        path = source / relative
        if relative_text in declared or path.is_symlink() or not path.is_file():
            raise CaptureError(f"legacy capture file is invalid: {relative_text}")
        if path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
            raise CaptureError(f"legacy capture file sha256 mismatch: {relative_text}")
        declared[relative_text] = entry

    scenes = []
    for scene_id in REQUIRED_SCENES:
        metadata_path = source / scene_id / "metadata.yaml"
        try:
            metadata = yaml.safe_load(metadata_path.read_bytes())
            information = metadata["rosbag2_bagfile_information"]
            duration_s = information["duration"]["nanoseconds"] / 1_000_000_000
            topic_entries = information["topics_with_message_count"]
            relative_files = information["relative_file_paths"]
        except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
            raise CaptureError(f"scene {scene_id} metadata is invalid: {exc}") from exc
        topics = {
            entry["topic_metadata"]["name"]: {
                "type": entry["topic_metadata"]["type"],
                "count": entry["message_count"],
            }
            for entry in topic_entries
        }
        scene_files = [{"path": f"{scene_id}/metadata.yaml"}]
        scene_files.extend({"path": f"{scene_id}/{name}"} for name in relative_files)
        scene = {
            "scene_id": scene_id,
            "role": "test" if scene_id == "scene-04-test" else "fit",
            "duration_s": duration_s,
            "stationary_windows": 1,
            "topics": topics,
            "tf_edges": ["base_link -> body", "body -> camera_front_link"],
            "files": scene_files,
        }
        _validate_scene(scene)
        if any(entry["path"] not in declared for entry in scene_files):
            raise CaptureError(f"scene {scene_id} has an undeclared storage file")
        scenes.append(scene)
    return finalize_capture(
        root,
        transfer["bundle_id"],
        scenes,
        devices,
        source=source,
        legacy_transfer_manifest_sha256=hashlib.sha256(content).hexdigest(),
    )


def finalize_cli(argv: list[str] | None = None) -> int:
    """Finalize a JSON scene summary produced by an external recorder."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.summary.read_text(encoding="utf-8"))
        result = finalize_capture(
            args.output,
            document["capture_id"],
            document["scenes"],
            document.get("devices", {}),
            source=args.summary.parent,
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, CaptureError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(result)
    return 0
