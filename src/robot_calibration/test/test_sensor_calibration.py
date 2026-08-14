import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

from robot_calibration.sensor_calibration import (
    CalibrationContractError,
    invert_transform,
    main,
    require_calibration_ready,
    resolve_sensor_calibration,
)

ARTIFACT_NAMES = ("base_to_front_camera", "base_to_mid360", "front_camera_intrinsics")
ROBOT_CALIBRATION_ROOT = Path(__file__).parents[1]


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _transform(child_frame: str, status: str = "approved") -> dict:
    return {
        "schema_version": "1.0",
        "calibration_version": "2026-08-12",
        "status": status,
        "device": {"name": "sensor", "serial": "sensor-1"},
        "transform": {
            "parent_frame": "base_link",
            "child_frame": child_frame,
            "translation": [0.1, -0.2, 0.3],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }


def _intrinsics(status: str = "approved") -> dict:
    return {
        "schema_version": "1.0",
        "calibration_version": "2026-08-12",
        "status": status,
        "device": {"name": "RealSense", "serial": "camera-1"},
        "camera_info": {
            "frame_id": "camera_front_optical_frame",
            "width": 640,
            "height": 480,
            "distortion_model": "plumb_bob",
            "d": [0.0] * 5,
            "k": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "p": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    }


def _config(tmp_path: Path, status: str = "approved") -> tuple[dict, dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "base_to_front_camera": tmp_path / "base_to_front_camera.yaml",
        "base_to_mid360": tmp_path / "base_to_mid360.yaml",
        "front_camera_intrinsics": tmp_path / "front_camera_intrinsics.yaml",
    }
    _write_yaml(paths["base_to_front_camera"], _transform("camera_front_optical_frame", status))
    _write_yaml(paths["base_to_mid360"], _transform("body", status))
    _write_yaml(paths["front_camera_intrinsics"], _intrinsics(status))
    return (
        {
            "sensor_calibration": {
                "artifacts": {name: str(path) for name, path in paths.items()},
                "expected_devices": {
                    "base_to_front_camera": "sensor-1",
                    "base_to_mid360": "sensor-1",
                    "front_camera_intrinsics": "camera-1",
                },
            }
        },
        paths,
    )


def test_non_opted_config_has_no_calibration_contract():
    assert resolve_sensor_calibration({"name": "legacy"}) is None


def test_all_approved_artifacts_are_ready_and_content_addressed(tmp_path):
    config, paths = _config(tmp_path)

    resolution = resolve_sensor_calibration(config)

    assert resolution["ready"] is True
    assert resolution["diagnostics"] == []
    assert tuple(resolution["artifacts"]) == ARTIFACT_NAMES
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    assert {name: item["sha256"] for name, item in resolution["artifacts"].items()} == hashes
    expected_bundle = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert resolution["bundle_sha256"] == expected_bundle
    require_calibration_ready(resolution)


def test_candidate_artifacts_are_valid_but_fail_production_gate(tmp_path):
    config, _ = _config(tmp_path, status="candidate")

    resolution = resolve_sensor_calibration(config)

    assert resolution["ready"] is False
    assert [item["state"] for item in resolution["artifacts"].values()] == ["candidate"] * 3
    assert [item["code"] for item in resolution["diagnostics"]] == ["CALIBRATION_NOT_APPROVED"] * 3
    with pytest.raises(CalibrationContractError, match="not ready"):
        require_calibration_ready(resolution)


def test_missing_artifact_is_reported_without_fabricating_a_bundle(tmp_path):
    config, paths = _config(tmp_path)
    paths["base_to_mid360"].unlink()

    resolution = resolve_sensor_calibration(config)

    assert resolution["ready"] is False
    assert resolution["bundle_sha256"] is None
    assert resolution["artifacts"]["base_to_mid360"]["state"] == "missing"
    assert resolution["diagnostics"] == [{"artifact": "base_to_mid360", "code": "CALIBRATION_FILE_MISSING"}]


def test_device_mismatch_and_invalid_transform_fail_closed(tmp_path):
    config, paths = _config(tmp_path)
    value = _transform("livox_frame")
    value["device"]["serial"] = "wrong"
    value["transform"]["rotation_xyzw"] = [0.0, 0.0, 0.0, 0.0]
    _write_yaml(paths["base_to_mid360"], value)

    resolution = resolve_sensor_calibration(config)

    assert resolution["artifacts"]["base_to_mid360"]["state"] == "invalid"
    assert [item["code"] for item in resolution["diagnostics"]] == [
        "CALIBRATION_DEVICE_MISMATCH",
        "CALIBRATION_FRAME_INVALID",
        "CALIBRATION_QUATERNION_INVALID",
    ]


def test_invert_transform_rotates_translation_and_returns_unit_quaternion():
    inverse = invert_transform(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
    )

    assert inverse["translation"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert inverse["rotation_xyzw"] == pytest.approx([0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)], abs=1e-12)


def test_cli_reports_ready_state_and_uses_stable_exit_codes(tmp_path, capsys):
    config, _ = _config(tmp_path)
    config_path = tmp_path / "robot.yaml"
    _write_yaml(config_path, {"robot": config})

    assert main([str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True

    candidate, _ = _config(tmp_path / "candidate", status="candidate")
    _write_yaml(config_path, {"robot": candidate})
    assert main([str(config_path)]) == 1
    assert json.loads(capsys.readouterr().out)["ready"] is False

    invalid, invalid_paths = _config(tmp_path / "invalid")
    invalid_paths["base_to_mid360"].write_text("not: [yaml", encoding="utf-8")
    _write_yaml(config_path, {"robot": invalid})
    assert main([str(config_path)]) == 2
    assert json.loads(capsys.readouterr().out)["artifacts"]["base_to_mid360"]["state"] == "invalid"

    _write_yaml(config_path, {"not_robot": {}})
    assert main([str(config_path)]) == 2
    assert "robot mapping" in capsys.readouterr().err


def test_checked_in_examples_are_valid_candidate_artifacts():
    example_dir = ROBOT_CALIBRATION_ROOT / "config" / "examples"
    config = {"sensor_calibration": {"artifacts": {name: str(example_dir / f"{name}.yaml") for name in ARTIFACT_NAMES}}}

    resolution = resolve_sensor_calibration(config)

    assert resolution["ready"] is False
    assert [item["state"] for item in resolution["artifacts"].values()] == ["candidate"] * 3
    assert [item["code"] for item in resolution["diagnostics"]] == ["CALIBRATION_NOT_APPROVED"] * 3
