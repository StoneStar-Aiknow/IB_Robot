import json
import subprocess
from pathlib import Path

import pytest

from robot_calibration import validation
from robot_calibration.validation import validate_artifact_archive, validation_lock


def test_validate_artifact_archive_accepts_single_user_archive(tmp_path):
    archive = tmp_path / "calib-1.candidate.tar"
    archive.write_bytes(b"placeholder")
    # Validation is deliberately delegated to the archive importer; this test
    # checks the user-facing path classification before ROS is started.
    assert validate_artifact_archive(archive) == archive


def test_validation_summary_has_no_production_activation(tmp_path):
    summary = tmp_path / "calibration_summary.json"
    summary.write_text(json.dumps({"status": "candidate"}), encoding="utf-8")
    assert json.loads(summary.read_text())["status"] == "candidate"


def test_validation_cli_exposes_only_archive_input():
    completed = subprocess.run(
        ["python3", "-c", "from robot_calibration.validation import main; main(['--help'])"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--input" in completed.stdout
    assert "--mount" not in completed.stdout
    assert "--output-topic" not in completed.stdout


def test_validation_owns_the_sensor_graph(tmp_path, monkeypatch):
    archive = tmp_path / "calib-1.candidate.tar"
    artifact = tmp_path / "base_to_front_camera.candidate.yaml"
    artifact.write_text("status: candidate\n", encoding="utf-8")
    import tarfile

    with tarfile.open(archive, "w") as target:
        target.add(artifact, arcname=artifact.name)
    mount = tmp_path / "mount.yaml"
    mount.write_text("translation_m: [0, 0, 0]\n", encoding="utf-8")
    stopped = []
    preview = object()
    sensor = object()
    monkeypatch.setattr(validation, "_start_capture_preview", lambda _log_path: preview)
    monkeypatch.setattr(validation, "_start_sensor_calibration", lambda _log_path: sensor)
    monkeypatch.setattr(validation, "_stop_owned_process", stopped.append)
    monkeypatch.setattr(validation, "start_viewer", lambda _mode, _log_path: None)
    monkeypatch.setattr(
        validation.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 3),
    )

    assert validation.run_validation(archive, mount=mount) == 3
    assert stopped == [preview, sensor]


def test_validation_starts_and_stops_the_sensor_graph(tmp_path, monkeypatch):
    archive = tmp_path / "calib-1.candidate.tar"
    artifact = tmp_path / "base_to_front_camera.candidate.yaml"
    artifact.write_text("status: candidate\n", encoding="utf-8")
    import tarfile

    with tarfile.open(archive, "w") as target:
        target.add(artifact, arcname=artifact.name)
    mount = tmp_path / "mount.yaml"
    mount.write_text("translation_m: [0, 0, 0]\n", encoding="utf-8")
    started = []
    stopped = []
    preview = object()
    sensor = object()
    monkeypatch.setattr(validation, "_start_capture_preview", lambda _log_path: preview)
    monkeypatch.setattr(validation, "_start_sensor_calibration", lambda _log_path: started.append(True) or sensor)
    monkeypatch.setattr(validation, "_stop_owned_process", stopped.append)
    monkeypatch.setattr(validation, "start_viewer", lambda _mode, _log_path: None)
    monkeypatch.setattr(
        validation.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 3),
    )

    assert validation.run_validation(archive, mount=mount) == 3
    assert started == [True]
    assert stopped == [preview, sensor]


def test_validation_uses_candidate_mount_from_archive(tmp_path, monkeypatch):
    archive = tmp_path / "calib-1.candidate.tar"
    artifact = tmp_path / "base_to_front_camera.candidate.yaml"
    artifact.write_text("status: candidate\n", encoding="utf-8")
    mount = tmp_path / "base_to_mid360.yaml"
    mount.write_text("calibration_version: candidate-mount\n", encoding="utf-8")
    import tarfile

    with tarfile.open(archive, "w") as target:
        target.add(artifact, arcname=artifact.name)
        target.add(mount, arcname=mount.name)

    captured = {}
    monkeypatch.setattr(validation, "_start_capture_preview", lambda _log_path: None)
    monkeypatch.setattr(validation, "_start_sensor_calibration", lambda _log_path: None)
    monkeypatch.setattr(validation, "_stop_owned_process", lambda _process: None)
    monkeypatch.setattr(validation, "start_viewer", lambda _mode, _log_path: None)

    def fake_run(command, **_kwargs):
        mount_path = Path(command[command.index("--mount") + 1])
        captured["mount"] = mount_path.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)

    assert validation.run_validation(archive) == 0
    assert captured["mount"] == "calibration_version: candidate-mount\n"


def test_validation_lock_rejects_a_second_instance(tmp_path):
    lock_path = tmp_path / "validate.lock"

    with (
        validation_lock(lock_path),
        pytest.raises(RuntimeError, match="已有标定验证正在运行"),
        validation_lock(lock_path),
    ):
        pass

    assert Path(lock_path).is_file()
