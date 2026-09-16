"""Validate content-addressed sensor calibration artifacts."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ARTIFACT_ORDER = ("base_to_front_camera", "base_to_mid360", "front_camera_intrinsics")
EXPECTED_FRAMES = {
    "base_to_front_camera": ("base_link", "camera_front_optical_frame"),
    "base_to_mid360": ("base_link", "body"),
}


class CalibrationContractError(RuntimeError):
    """Raised when calibration cannot be used for production."""

    def __init__(self, message: str, resolution: dict[str, Any] | None = None):
        super().__init__(message)
        self.resolution = resolution


def _finite_vector(value: object, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _normalized_quaternion(value: object) -> list[float] | None:
    quaternion = _finite_vector(value, 4)
    if quaternion is None:
        return None
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 1e-12:
        return None
    return [item / norm for item in quaternion]


def invert_transform(translation: object, rotation_xyzw: object) -> dict[str, list[float]]:
    """Return the inverse of a rigid transform."""
    vector = _finite_vector(translation, 3)
    quaternion = _normalized_quaternion(rotation_xyzw)
    if vector is None or quaternion is None:
        raise ValueError("transform must contain a finite translation and non-zero quaternion")
    x, y, z, w = quaternion
    inverse_rotation = [-x, -y, -z, w]
    vx, vy, vz = (-vector[0], -vector[1], -vector[2])
    tx = 2.0 * (inverse_rotation[1] * vz - inverse_rotation[2] * vy)
    ty = 2.0 * (inverse_rotation[2] * vx - inverse_rotation[0] * vz)
    tz = 2.0 * (inverse_rotation[0] * vy - inverse_rotation[1] * vx)
    inverse_translation = [
        vx + inverse_rotation[3] * tx + inverse_rotation[1] * tz - inverse_rotation[2] * ty,
        vy + inverse_rotation[3] * ty + inverse_rotation[2] * tx - inverse_rotation[0] * tz,
        vz + inverse_rotation[3] * tz + inverse_rotation[0] * ty - inverse_rotation[1] * tx,
    ]
    return {"translation": inverse_translation, "rotation_xyzw": inverse_rotation}


def _diagnostic(artifact: str, code: str) -> dict[str, str]:
    return {"artifact": artifact, "code": code}


def _validate_common(name: str, value: dict[str, Any], expected_serial: object) -> list[dict[str, str]]:
    diagnostics = []
    if value.get("schema_version") != "1.0":
        diagnostics.append(_diagnostic(name, "CALIBRATION_SCHEMA_UNSUPPORTED"))
    version = value.get("calibration_version")
    if not isinstance(version, str) or not version.strip():
        diagnostics.append(_diagnostic(name, "CALIBRATION_VERSION_MISSING"))
    if value.get("status") not in {"candidate", "approved"}:
        diagnostics.append(_diagnostic(name, "CALIBRATION_STATUS_INVALID"))
    device = value.get("device")
    if not isinstance(device, dict) or not isinstance(device.get("name"), str) or not device["name"].strip():
        diagnostics.append(_diagnostic(name, "CALIBRATION_DEVICE_INVALID"))
    serial = device.get("serial") if isinstance(device, dict) else None
    if expected_serial and serial != expected_serial:
        diagnostics.append(_diagnostic(name, "CALIBRATION_DEVICE_MISMATCH"))
    return diagnostics


def _validate_transform(name: str, value: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = []
    transform = value.get("transform")
    parent, child = EXPECTED_FRAMES[name]
    if not isinstance(transform, dict) or (
        transform.get("parent_frame") != parent or transform.get("child_frame") != child
    ):
        diagnostics.append(_diagnostic(name, "CALIBRATION_FRAME_INVALID"))
    if not isinstance(transform, dict) or _finite_vector(transform.get("translation"), 3) is None:
        diagnostics.append(_diagnostic(name, "CALIBRATION_VALUE_INVALID"))
    if not isinstance(transform, dict) or _normalized_quaternion(transform.get("rotation_xyzw")) is None:
        diagnostics.append(_diagnostic(name, "CALIBRATION_QUATERNION_INVALID"))
    return diagnostics


def _validate_intrinsics(name: str, value: dict[str, Any]) -> list[dict[str, str]]:
    info = value.get("camera_info")
    if not isinstance(info, dict):
        return [_diagnostic(name, "CALIBRATION_INTRINSICS_INVALID")]
    valid = info.get("frame_id") == "camera_front_optical_frame"
    valid = valid and isinstance(info.get("width"), int) and info["width"] > 0
    valid = valid and isinstance(info.get("height"), int) and info["height"] > 0
    valid = valid and info.get("distortion_model") == "plumb_bob"
    valid = valid and _finite_vector(info.get("d"), 5) is not None
    valid = valid and _finite_vector(info.get("k"), 9) is not None
    valid = valid and _finite_vector(info.get("r"), 9) is not None
    valid = valid and _finite_vector(info.get("p"), 12) is not None
    return [] if valid else [_diagnostic(name, "CALIBRATION_INTRINSICS_INVALID")]


def _load_artifact(name: str, path: Path, expected_serial: object) -> dict[str, Any]:
    if not path.is_file():
        return {
            "state": "missing",
            "path": str(path),
            "sha256": None,
            "size": None,
            "data": None,
            "diagnostics": [_diagnostic(name, "CALIBRATION_FILE_MISSING")],
        }
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    try:
        value = yaml.safe_load(content) or {}
    except yaml.YAMLError:
        value = None
    if not isinstance(value, dict):
        diagnostics = [_diagnostic(name, "CALIBRATION_YAML_INVALID")]
        state = "invalid"
    else:
        diagnostics = _validate_common(name, value, expected_serial)
        diagnostics.extend(
            _validate_intrinsics(name, value) if name == "front_camera_intrinsics" else _validate_transform(name, value)
        )
        state = value.get("status") if not diagnostics else "invalid"
    return {
        "state": state,
        "path": str(path),
        "sha256": digest,
        "size": len(content),
        "data": value,
        "diagnostics": diagnostics,
    }


def resolve_sensor_calibration(robot_config: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve an opted-in calibration contract without mutating the robot config."""
    contract = robot_config.get("sensor_calibration")
    if contract is None:
        return None
    if not isinstance(contract, dict) or not isinstance(contract.get("artifacts"), dict):
        raise CalibrationContractError("sensor_calibration.artifacts must be a mapping")
    expected_devices = contract.get("expected_devices", {})
    artifacts = {}
    diagnostics = []
    for name in ARTIFACT_ORDER:
        configured_path = contract["artifacts"].get(name, "")
        artifact = _load_artifact(name, Path(str(configured_path)).expanduser(), expected_devices.get(name))
        artifacts[name] = artifact
        diagnostics.extend(artifact["diagnostics"])
        if artifact["state"] == "candidate":
            diagnostics.append(_diagnostic(name, "CALIBRATION_NOT_APPROVED"))
    valid_artifacts = all(item["state"] in {"candidate", "approved"} for item in artifacts.values())
    hashes = {name: item["sha256"] for name, item in artifacts.items()}
    bundle_sha256 = None
    if valid_artifacts:
        canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        bundle_sha256 = hashlib.sha256(canonical).hexdigest()
    return {
        "schema_version": "1.0",
        "ready": all(item["state"] == "approved" for item in artifacts.values()),
        "bundle_sha256": bundle_sha256,
        "artifacts": artifacts,
        "diagnostics": diagnostics,
    }


def require_calibration_ready(resolution: dict[str, Any] | None) -> None:
    """Fail closed unless a complete approved bundle is available."""
    if resolution is None or not resolution.get("ready", False):
        raise CalibrationContractError("sensor calibration is not ready", resolution)


def main(argv: list[str] | None = None) -> int:
    """Validate calibration artifacts declared by a robot configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_config", type=Path)
    args = parser.parse_args(argv)
    try:
        document = yaml.safe_load(args.robot_config.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("robot"), dict):
            raise ValueError("robot configuration must contain a robot mapping")
        resolution = resolve_sensor_calibration(document["robot"])
        if resolution is None:
            raise ValueError("robot configuration does not declare sensor_calibration")
    except (OSError, ValueError, yaml.YAMLError, CalibrationContractError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(resolution, sort_keys=True, separators=(",", ":")))
    if any(item["state"] == "invalid" for item in resolution["artifacts"].values()):
        return 2
    return 0 if resolution["ready"] else 1
