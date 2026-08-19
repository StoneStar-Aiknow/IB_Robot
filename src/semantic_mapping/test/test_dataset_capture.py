import hashlib
import json
import signal
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from semantic_mapping import dataset_capture


def _write_map(root: Path) -> None:
    (root / "map").mkdir(parents=True, exist_ok=True)
    (root / "map" / "map.pgm").write_bytes(b"P5\n1 1\n255\n\0")
    (root / "map" / "map.yaml").write_text("image: map.pgm\nresolution: 0.05\n", encoding="utf-8")


def _topic(name, message_type, count):
    return {"topic_metadata": {"name": name, "type": message_type}, "message_count": count}


def _required_topics():
    return [
        _topic(name, message_type, 10)
        for name, message_type in dataset_capture.SEMANTIC_RECORDED_TOPIC_TYPES.items()
        if name in dataset_capture.SEMANTIC_REQUIRED_NONZERO_TOPICS
    ]


def _write_bag(root: Path, splits=("bag_0.mcap", "bag_1.mcap"), topics=None, *, with_file_times=True) -> None:
    bag_dir = root / "bag"
    bag_dir.mkdir(parents=True, exist_ok=True)
    for name in splits:
        (bag_dir / name).write_bytes(name.encode())
    information = {
        "relative_file_paths": list(splits),
        "starting_time": {"nanoseconds_since_epoch": 100},
        "duration": {"nanoseconds": 400},
        "topics_with_message_count": topics or _required_topics(),
    }
    if with_file_times:
        information["files"] = [
            {
                "path": name,
                "starting_time": {"nanoseconds_since_epoch": 100 + index * 200},
                "duration": {"nanoseconds": 200},
                "message_count": 10,
            }
            for index, name in enumerate(splits)
        ]
    (bag_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"rosbag2_bagfile_information": information}),
        encoding="utf-8",
    )


def _camera_info():
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="camera_front_optical_frame"),
        width=640,
        height=480,
        distortion_model="plumb_bob",
        d=[0.1, 0.2, 0.0, 0.0, 0.0],
        k=[1.0] * 9,
        r=[1.0] * 9,
        p=[1.0] * 12,
    )


class _CameraInfoReader:
    def __init__(self, message=None):
        self.message = message or _camera_info()

    def messages(self, topics):
        topic = next(iter(topics))
        yield topic, self.message, 1


class _Completed:
    returncode = 0


class _Failed:
    returncode = 7


def test_validate_mcap_metadata_requires_every_actual_split(tmp_path):
    _write_bag(tmp_path)

    assert dataset_capture.validate_mcap_metadata(tmp_path / "bag")["relative_file_paths"] == [
        "bag_0.mcap",
        "bag_1.mcap",
    ]
    metadata = yaml.safe_load((tmp_path / "bag" / "metadata.yaml").read_text(encoding="utf-8"))
    metadata["rosbag2_bagfile_information"]["relative_file_paths"] = ["bag_1.mcap"]
    (tmp_path / "bag" / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="MCAP split list"):
        dataset_capture.validate_mcap_metadata(tmp_path / "bag")


def test_validate_mcap_metadata_uses_actual_topic_type_count_and_time_coverage(tmp_path):
    topics = _required_topics() + [
        _topic("/cmd_vel", "geometry_msgs/msg/Twist", 0),
    ]
    _write_bag(tmp_path, topics=topics)

    result = dataset_capture.validate_mcap_metadata(tmp_path / "bag")

    actual = {item["name"]: item for item in result["topics"]}
    assert actual["/livox/lidar"] == {
        "name": "/livox/lidar",
        "type": "livox_ros_driver2/msg/CustomMsg",
        "message_count": 10,
    }
    assert result["topic_validation"]["required_nonzero"] == sorted(dataset_capture.SEMANTIC_REQUIRED_NONZERO_TOPICS)
    assert result["topic_validation"]["reported_optional"]["/cmd_vel"] == 0
    assert result["topic_validation"]["missing_optional"] == ["/wheel/odom"]
    assert result["time_coverage"]["status"] == "validated"
    assert result["time_coverage"]["files_end_ns"] == 500


def test_validate_mcap_metadata_rejects_missing_or_empty_required_topic(tmp_path):
    topics = [item for item in _required_topics() if item["topic_metadata"]["name"] != "/scan"]
    topics.append(_topic("/livox/lidar", "livox_ros_driver2/msg/CustomMsg", 0))
    _write_bag(tmp_path, topics=topics)

    with pytest.raises(ValueError, match="required recorded topics"):
        dataset_capture.validate_mcap_metadata(tmp_path / "bag")


def test_validate_mcap_metadata_rejects_required_topic_type_mismatch(tmp_path):
    topics = _required_topics()
    lidar = next(item for item in topics if item["topic_metadata"]["name"] == "/livox/lidar")
    lidar["topic_metadata"]["type"] = "sensor_msgs/msg/PointCloud2"
    _write_bag(tmp_path, topics=topics)

    with pytest.raises(ValueError, match="wrong_types"):
        dataset_capture.validate_mcap_metadata(tmp_path / "bag")


def test_validate_mcap_metadata_rejects_top_level_time_that_does_not_cover_splits(tmp_path):
    _write_bag(tmp_path)
    metadata_path = tmp_path / "bag" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["rosbag2_bagfile_information"]["duration"]["nanoseconds"] = 399
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="time range"):
        dataset_capture.validate_mcap_metadata(tmp_path / "bag")


def test_validate_mcap_metadata_rejects_partial_split_time_coverage(tmp_path):
    _write_bag(tmp_path)
    metadata_path = tmp_path / "bag" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["rosbag2_bagfile_information"]["files"] = metadata["rosbag2_bagfile_information"]["files"][:1]
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="coverage paths"):
        dataset_capture.validate_mcap_metadata(tmp_path / "bag")


@pytest.mark.parametrize(
    "covered_paths",
    [
        ["bag_0.mcap", "./bag_0.mcap"],
        ["bag_0.mcap", "unknown.mcap"],
    ],
)
def test_validate_mcap_metadata_rejects_duplicate_or_mismatched_split_time_coverage(tmp_path, covered_paths):
    _write_bag(tmp_path)
    metadata_path = tmp_path / "bag" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    for item, path in zip(metadata["rosbag2_bagfile_information"]["files"], covered_paths, strict=True):
        item["path"] = path
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="coverage paths"):
        dataset_capture.validate_mcap_metadata(tmp_path / "bag")


def test_validate_mcap_metadata_records_unavailable_file_time_coverage(tmp_path):
    _write_bag(tmp_path, with_file_times=False)

    result = dataset_capture.validate_mcap_metadata(tmp_path / "bag")

    assert result["time_coverage"] == {"status": "unavailable", "reason": "per-file time data absent"}


def test_validate_mcap_metadata_requires_top_level_time_when_split_times_exist(tmp_path):
    _write_bag(tmp_path)
    metadata_path = tmp_path / "bag" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    del metadata["rosbag2_bagfile_information"]["starting_time"]
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="top-level"):
        dataset_capture.validate_mcap_metadata(tmp_path / "bag")


def test_validate_mcap_metadata_rejects_empty_bag(tmp_path):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"rosbag2_bagfile_information": {"relative_file_paths": []}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="MCAP split list"):
        dataset_capture.validate_mcap_metadata(bag_dir)


def test_snapshot_uses_pinned_mount_and_recorded_camera_info_but_allows_missing_camera(tmp_path):
    _write_bag(tmp_path, ("bag_0.mcap",))
    mount = tmp_path / "mount.yaml"
    mount.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "status": "provisional",
                "parent_frame": "base_link",
                "body_frame": "body",
                "translation_m": [-0.08, 0.0, 0.2],
                "rpy_deg": [0.0, 0.0, 90.0],
            }
        ),
        encoding="utf-8",
    )
    sources = {
        "base_to_mid360": {
            "status": "provisional",
            "source": "/mutable/mount.yaml",
            "snapshot": str(mount),
            "sha256": hashlib.sha256(mount.read_bytes()).hexdigest(),
        },
        "base_to_front_camera": {"status": "missing", "source": "/missing/camera.yaml"},
    }

    snapshot = dataset_capture.build_calibration_snapshot(
        tmp_path / "bag",
        calibration_sources=sources,
        camera_info_topic="/camera/front/camera_info",
        reader_factory=lambda _path: _CameraInfoReader(),
    )

    assert snapshot["status"] == "calibration_incomplete"
    assert snapshot["artifacts"]["base_to_front_camera"]["status"] == "missing"
    mount_artifact = snapshot["artifacts"]["base_to_mid360"]
    assert mount_artifact["status"] == "provisional"
    assert mount_artifact["data"]["parent_frame"] == "base_link"
    assert mount_artifact["data"]["child_frame"] == "body"
    assert mount_artifact["data"]["translation"] == [-0.08, 0.0, 0.2]
    intrinsics = snapshot["artifacts"]["front_camera_intrinsics"]
    assert intrinsics["status"] == "recorded"
    assert intrinsics["data"]["frame_id"] == "camera_front_optical_frame"
    assert intrinsics["data"]["width"] == 640
    assert intrinsics["data"]["K"] == [1.0] * 9


def test_snapshot_records_present_approved_camera_artifact(tmp_path):
    _write_bag(tmp_path, ("bag_0.mcap",))
    mount = tmp_path / "mount.yaml"
    mount.write_text(
        'schema_version: "1.0"\nstatus: provisional\nparent_frame: base_link\n'
        "body_frame: body\ntranslation_m: [-0.08, 0, 0.2]\nrpy_deg: [0, 0, 90]\n",
        encoding="utf-8",
    )
    camera = tmp_path / "base_to_front_camera.yaml"
    camera.write_text(
        "status: approved\ntransform:\n  parent_frame: base_link\n"
        "  child_frame: camera_front_optical_frame\n  translation: [0.1, 0, 0.2]\n"
        "  rotation_xyzw: [0, 0, 0, 1]\n",
        encoding="utf-8",
    )

    snapshot = dataset_capture.build_calibration_snapshot(
        tmp_path / "bag",
        calibration_sources={
            "base_to_mid360": {
                "status": "provisional",
                "source": str(mount),
                "snapshot": str(mount),
                "sha256": hashlib.sha256(mount.read_bytes()).hexdigest(),
            },
            "base_to_front_camera": {
                "status": "approved",
                "source": str(camera),
                "snapshot": str(camera),
                "sha256": hashlib.sha256(camera.read_bytes()).hexdigest(),
            },
        },
        camera_info_topic="/camera/front/camera_info",
        reader_factory=lambda _path: _CameraInfoReader(),
    )

    artifact = snapshot["artifacts"]["base_to_front_camera"]
    assert snapshot["status"] == "complete"
    assert artifact["status"] == "approved"
    assert artifact["sha256"] == hashlib.sha256(camera.read_bytes()).hexdigest()


def test_finalize_stops_recorder_saves_map_reindexes_then_generates_artifacts(tmp_path, monkeypatch):
    session_root = tmp_path / "semantic_session"
    session_root.mkdir()
    state_file = tmp_path / "current.json"
    mount = tmp_path / "mount.yaml"
    mount.write_text(
        'schema_version: "1.0"\nstatus: provisional\nparent_frame: base_link\n'
        "body_frame: body\ntranslation_m: [-0.08, 0, 0.2]\nrpy_deg: [0, 0, 90]\n",
        encoding="utf-8",
    )
    calibration_dir = session_root / "bag" / "calibration_sources"
    calibration_dir.mkdir(parents=True)
    pinned_mount = calibration_dir / "base_to_mid360.yaml"
    pinned_mount.write_bytes(mount.read_bytes())
    state = {
        "session_id": "20260815_120000",
        "profile": "lekiwi_semantic_capture",
        "session_root": str(session_root),
        "supervisor_pid": 4242,
        "status": "recording",
        "calibration_sources": {
            "base_to_mid360": {
                "status": "provisional",
                "source": str(mount),
                "snapshot": str(pinned_mount),
                "sha256": hashlib.sha256(pinned_mount.read_bytes()).hexdigest(),
            },
            "base_to_front_camera": {"status": "missing", "source": str(tmp_path / "missing-camera.yaml")},
        },
        "camera_info_topic": "/camera/front/camera_info",
        "topics": ["/camera/front/image_raw", "/tf"],
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")
    events = []

    def run(command, check=False):
        events.append(tuple(command))
        if command[:4] == ["ros2", "run", "robot_navigation", "save_lidar_map"]:
            _write_map(session_root)
        elif command[:4] == ["ros2", "bag", "reindex", "--storage"]:
            _write_bag(session_root)
        return _Completed()

    def kill(pid, sent_signal):
        events.append(("kill", pid, sent_signal))
        state["status"] = "recorded"
        state_file.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(dataset_capture.os, "kill", kill)
    monkeypatch.setattr(dataset_capture.time, "sleep", lambda _seconds: None)

    archive = dataset_capture.finalize_dataset(
        state_file,
        command_runner=run,
        reader_factory=lambda _path: _CameraInfoReader(),
    )

    assert events[0] == ("kill", 4242, signal.SIGINT)
    assert events[1][:4] == ("ros2", "run", "robot_navigation", "save_lidar_map")
    assert events[2][:5] == ("ros2", "bag", "reindex", "--storage", "mcap")
    assert (session_root / "bag" / "calibration_snapshot.json").is_file()
    assert (session_root / "manifest.json").is_file()
    assert (session_root / "SHA256SUMS").is_file()
    assert (session_root / "README.md").is_file()
    assert archive == session_root.with_suffix(".tar.gz")
    manifest = json.loads((session_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "calibration_incomplete"
    assert manifest["navigation_map"]["status"] == "pending_validation"
    assert "Navigation map promotion status: `pending_validation`" in (session_root / "README.md").read_text(
        encoding="utf-8"
    )
    topics = {item["name"]: item for item in manifest["topics"]}
    assert topics["/camera/front/aligned_depth_to_color/image_raw"] == {
        "message_count": 10,
        "name": "/camera/front/aligned_depth_to_color/image_raw",
        "type": "sensor_msgs/msg/Image",
    }


def test_finalize_returns_map_failure_after_recorder_stop_without_reindex(tmp_path, monkeypatch):
    session_root = tmp_path / "semantic_session"
    session_root.mkdir()
    state_file = tmp_path / "current.json"
    state = {
        "session_root": str(session_root),
        "supervisor_pid": 4242,
        "status": "recording",
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")
    events = []

    def run(command, check=False):
        events.append(tuple(command))
        return _Failed()

    def kill(pid, sent_signal):
        events.append(("kill", pid, sent_signal))
        state["status"] = "recorded"
        state_file.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(dataset_capture.os, "kill", kill)
    monkeypatch.setattr(dataset_capture.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="return code 7"):
        dataset_capture.finalize_dataset(state_file, command_runner=run)

    assert events[0] == ("kill", 4242, signal.SIGINT)
    assert events[1][:4] == ("ros2", "run", "robot_navigation", "save_lidar_map")
    assert not any(event[:2] == ("ros2", "bag") for event in events[2:])


def test_deterministic_tar_gz_layout_and_checksums(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _write_bag(root, ("bag_0.mcap",))
    _write_map(root)
    (root / "bag" / "calibration_snapshot.json").write_text("{}\n", encoding="utf-8")
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "README.md").write_text("dataset\n", encoding="utf-8")
    dataset_capture.write_checksums(root)
    dataset_capture.verify_checksums(root)

    first = dataset_capture.create_deterministic_tar(root)
    first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    second = dataset_capture.create_deterministic_tar(root)

    assert second == root.with_suffix(".tar.gz")
    assert hashlib.sha256(second.read_bytes()).hexdigest() == first_hash
    with tarfile.open(second, "r:gz") as archive:
        names = archive.getnames()
    assert names == sorted(names)
    assert "bag/bag_0.mcap" in names
    assert "bag/metadata.yaml" in names
    assert "bag/calibration_snapshot.json" in names
    assert "map/map.yaml" in names
    assert "map/map.pgm" in names
    assert "manifest.json" in names
    assert "SHA256SUMS" in names
    assert "README.md" in names

    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest

    (root / "README.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum verification failed"):
        dataset_capture.verify_checksums(root)


def test_refresh_navigation_map_archive_records_promotion_result(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _write_bag(root, ("bag_0.mcap",))
    _write_map(root)
    (root / "bag" / "calibration_snapshot.json").write_text("{}\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "navigation_map": {
                    "status": "pending_validation",
                    "source_files": ["map/map.yaml", "map/map.pgm"],
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "README.md").write_text("Navigation map promotion status: `pending_validation`\n", encoding="utf-8")
    dataset_capture.write_checksums(root)
    target = tmp_path / "active" / "map"
    target.parent.mkdir()
    target.with_suffix(".yaml").write_bytes((root / "map" / "map.yaml").read_bytes())
    target.with_suffix(".pgm").write_bytes((root / "map" / "map.pgm").read_bytes())

    archive = dataset_capture._refresh_navigation_map_archive(
        root,
        "promoted",
        source_prefix=root / "map" / "map",
        target_prefix=target,
    )

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["navigation_map"]["status"] == "promoted"
    assert manifest["navigation_map"]["source_and_target_match"] is True
    assert archive == root.with_suffix(".tar.gz")
    dataset_capture.verify_checksums(root)


def test_tar_rejects_dataset_without_saved_map(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _write_bag(root, ("bag_0.mcap",))
    (root / "bag" / "calibration_snapshot.json").write_text("{}\n", encoding="utf-8")
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "README.md").write_text("dataset\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML/PGM pair"):
        dataset_capture.write_checksums(root)


def test_save_semantic_map_cli_finalizes_then_validates_dataset(tmp_path, monkeypatch, capsys):
    session_root = tmp_path / "session"
    archive = session_root.with_suffix(".tar.gz")
    archive.write_bytes(b"compressed archive")
    events = []

    def finalize(state_file, **_kwargs):
        events.append(("finalize", state_file))
        return archive

    def smoke(path, **_kwargs):
        events.append(("smoke", path))
        return {"calibration_status": "complete", "geometry_ready": True}

    def verify(path):
        events.append(("verify", path))

    def promote(path):
        events.append(("promote", path))

    monkeypatch.setattr(dataset_capture, "finalize_dataset", finalize)
    monkeypatch.setattr(dataset_capture, "smoke_dataset", smoke, raising=False)
    monkeypatch.setattr(dataset_capture, "verify_checksums", verify, raising=False)
    monkeypatch.setattr(dataset_capture, "promote_navigation_map", promote)
    monkeypatch.setattr(
        dataset_capture,
        "_refresh_navigation_map_archive",
        lambda session_root, status, **_kwargs: events.append(("refresh", session_root, status)) or archive,
    )

    assert dataset_capture.main([]) == 0
    assert events == [
        ("finalize", Path("~/.ros/ibrobot/semantic_mapping/current.json")),
        ("verify", session_root),
        ("smoke", session_root),
        ("promote", session_root / "map" / "map"),
        ("refresh", session_root, "promoted"),
        ("verify", session_root),
    ]
    assert "Semantic map saved" in capsys.readouterr().out


def test_save_semantic_map_cli_fails_when_offline_validation_fails(tmp_path, monkeypatch):
    state_file = tmp_path / "current.json"
    state_file.write_text(json.dumps({"status": "finalized"}), encoding="utf-8")
    archive = (tmp_path / "session").with_suffix(".tar.gz")
    monkeypatch.setattr(dataset_capture, "finalize_dataset", lambda *_args, **_kwargs: archive)
    monkeypatch.setattr(dataset_capture, "verify_checksums", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        dataset_capture,
        "smoke_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no synchronized RGB-D frame")),
        raising=False,
    )
    promoted = []
    monkeypatch.setattr(dataset_capture, "promote_navigation_map", lambda path: promoted.append(path))

    assert dataset_capture.main(["--session-file", str(state_file)]) == 1
    assert promoted == []
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["status"] == "semantic_map_offline_validation_failed"
    assert state["failure_stage"] == "offline_validation"


def test_save_semantic_map_cli_records_navigation_map_promotion_failure(tmp_path, monkeypatch):
    state_file = tmp_path / "current.json"
    state_file.write_text(json.dumps({"status": "finalized"}), encoding="utf-8")
    archive = (tmp_path / "session").with_suffix(".tar.gz")
    archive.write_bytes(b"compressed archive")
    monkeypatch.setattr(dataset_capture, "finalize_dataset", lambda *_args, **_kwargs: archive)
    monkeypatch.setattr(dataset_capture, "verify_checksums", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        dataset_capture,
        "smoke_dataset",
        lambda *_args, **_kwargs: {"calibration_status": "complete", "geometry_ready": True},
    )
    monkeypatch.setattr(
        dataset_capture,
        "promote_navigation_map",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("destination is not writable")),
    )

    assert dataset_capture.main(["--session-file", str(state_file)]) == 1

    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["status"] == "navigation_map_promotion_failed"
    assert state["promotion_error"] == "destination is not writable"


def test_package_exposes_one_semantic_map_save_command():
    setup_text = (Path(__file__).parents[1] / "setup.py").read_text(encoding="utf-8")

    assert "save_semantic_map = semantic_mapping.dataset_capture:main" in setup_text
    assert "save_semantic_dataset =" not in setup_text
    assert "semantic_dataset_smoke =" not in setup_text
