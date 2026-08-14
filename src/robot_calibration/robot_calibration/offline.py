"""Joint FAST-Calib solve and candidate runtime artifact generation."""

import hashlib
import itertools
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

FITTING_SCENES = ("scene-01", "scene-02", "scene-03")
TEST_SCENE = "scene-04-test"
REQUIRED_SCENES = (*FITTING_SCENES, TEST_SCENE)
SOLVER_REPOSITORY = "https://github.com/TommyBrownson/FAST-Calib_Ros2.git"
SOLVER_COMMIT = "7747dfc6109c04b4bf81d2e3661e41626c8392e1"
PATCH_DIFF_SHA256 = "4192e3430e1ddc3d5a7de6fdeb4d1b6a07d56af1f8359e28810e5f25fdfd5503"
_OBSERVATION_KEYS = {
    "schema_version",
    "scene_id",
    "source_frame",
    "target_frame",
    "camera_centers",
    "lidar_centers",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _centers(value: object, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (4, 3) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain exactly four finite XYZ centers")
    return array


def _load_observation(path: Path, expected_scene: str) -> tuple[np.ndarray, np.ndarray, str]:
    content = path.read_bytes()
    value = yaml.safe_load(content)
    if not isinstance(value, dict) or set(value) != _OBSERVATION_KEYS or value.get("schema_version") != "1.0":
        raise ValueError(f"{expected_scene} observation must use the closed schema_version 1.0 contract")
    if value.get("scene_id") != expected_scene:
        raise ValueError(f"observation scene_id must equal {expected_scene}")
    if value.get("source_frame") != "body" or value.get("target_frame") != "camera_front_optical_frame":
        raise ValueError(f"{expected_scene} observation frame contract is invalid")
    return (
        _centers(value.get("camera_centers"), f"{expected_scene} camera_centers"),
        _centers(value.get("lidar_centers"), f"{expected_scene} lidar_centers"),
        hashlib.sha256(content).hexdigest(),
    )


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    return rotation, target_center - rotation @ source_center


def _rmse(source: np.ndarray, target: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> float:
    residual = (rotation @ source.T).T + translation - target
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def solve_joint_calibration(
    *,
    observations: dict[str, str | Path],
    output: str | Path,
    report: str | Path,
    max_training_rmse_m: float,
    max_test_rmse_m: float,
    max_baseline_m: float,
    min_correspondence_margin_m: float,
) -> tuple[dict, dict]:
    """Fit three scenes and evaluate, but never fit, the independent test scene."""
    if set(observations) != set(REQUIRED_SCENES):
        raise ValueError("observations must contain exactly the required 3+1 scenes")
    thresholds = {
        "max_training_rmse_m": max_training_rmse_m,
        "max_test_rmse_m": max_test_rmse_m,
        "max_baseline_m": max_baseline_m,
        "min_correspondence_margin_m": min_correspondence_margin_m,
    }
    if any(not math.isfinite(value) or value <= 0 for value in thresholds.values()):
        raise ValueError("all solve thresholds must be positive and finite")

    loaded = {}
    inputs = {}
    for scene in REQUIRED_SCENES:
        path = Path(observations[scene]).expanduser().absolute()
        camera, lidar, digest = _load_observation(path, scene)
        loaded[scene] = (camera, lidar)
        inputs[scene] = {"path": str(path), "sha256": digest}

    permutations = tuple(itertools.permutations(range(4)))
    candidates = []
    for choices in itertools.product(permutations, repeat=len(FITTING_SCENES)):
        lidar = np.concatenate(
            [loaded[scene][1][list(order)] for scene, order in zip(FITTING_SCENES, choices, strict=True)]
        )
        camera = np.concatenate([loaded[scene][0] for scene in FITTING_SCENES])
        rotation, translation = _rigid_transform(lidar, camera)
        rmse = _rmse(lidar, camera, rotation, translation)
        candidates.append(((round(rmse, 9), float(np.linalg.norm(translation))), rotation, translation, choices))
    candidates.sort(key=lambda item: item[0])
    _, rotation, translation, choices = candidates[0]
    next_best_rmse = candidates[1][0][0]
    correspondence_margin = next_best_rmse - candidates[0][0][0]
    scene_rmse = {
        scene: _rmse(loaded[scene][1][list(order)], loaded[scene][0], rotation, translation)
        for scene, order in zip(FITTING_SCENES, choices, strict=True)
    }
    training_rmse = float(np.sqrt(np.mean(np.square(list(scene_rmse.values())))))
    test_camera, test_lidar = loaded[TEST_SCENE]
    test_order, test_rmse = min(
        ((order, _rmse(test_lidar[list(order)], test_camera, rotation, translation)) for order in permutations),
        key=lambda item: item[1],
    )
    baseline = float(np.linalg.norm(translation))
    for actual, limit, label in (
        (training_rmse, max_training_rmse_m, "training joint RMSE"),
        (test_rmse, max_test_rmse_m, "test RMSE"),
        (baseline, max_baseline_m, "sensor baseline"),
    ):
        if actual > limit:
            raise ValueError(f"{label} {actual:.6f} m exceeds {limit:.6f} m")
    if correspondence_margin < min_correspondence_margin_m:
        raise ValueError(
            f"correspondence margin {correspondence_margin:.6f} m is below {min_correspondence_margin_m:.6f} m"
        )

    result = {
        "schema_version": "1.0",
        "source_frame": "body",
        "target_frame": "camera_front_optical_frame",
        "transform_direction": "source_to_target",
        "translation_unit": "m",
        "rotation_convention": "active",
        "point_mapping": "p_target = R * p_source + t",
        "rotation_matrix": rotation.tolist(),
        "translation": translation.tolist(),
    }
    report_value = {
        "schema_version": "1.0",
        "status": "candidate",
        "fitting_scenes": list(FITTING_SCENES),
        "test_scene": TEST_SCENE,
        "training_joint_rmse_m": training_rmse,
        "fitting_scene_rmse_m": scene_rmse,
        "test_rmse_m": test_rmse,
        "sensor_baseline_m": baseline,
        "next_best_training_rmse_m": next_best_rmse,
        "correspondence_margin_m": correspondence_margin,
        "correspondence_indices": {
            **{scene: list(order) for scene, order in zip(FITTING_SCENES, choices, strict=True)},
            TEST_SCENE: list(test_order),
        },
        "thresholds": thresholds,
        "inputs": inputs,
    }
    result_content = yaml.safe_dump(result, sort_keys=False).encode()
    report_value["result_sha256"] = hashlib.sha256(result_content).hexdigest()
    report_content = json.dumps(report_value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    output_path, report_path = Path(output), Path(report)
    if output_path == report_path:
        raise ValueError("output and report paths must be different")
    _exclusive_write(output_path, result_content)
    try:
        _exclusive_write(report_path, report_content)
    except OSError:
        output_path.unlink(missing_ok=True)
        raise
    return result, report_value


def _matrix_from_quaternion(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1):
        raise ValueError("mount quaternion must be finite and normalized")
    x, y, z, w = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quaternion_from_matrix(matrix: np.ndarray) -> list[float]:
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.array(
            [
                [
                    matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
                    matrix[1, 0] + matrix[0, 1],
                    matrix[2, 0] + matrix[0, 2],
                    matrix[2, 1] - matrix[1, 2],
                ],
                [
                    matrix[1, 0] + matrix[0, 1],
                    matrix[1, 1] - matrix[0, 0] - matrix[2, 2],
                    matrix[2, 1] + matrix[1, 2],
                    matrix[0, 2] - matrix[2, 0],
                ],
                [
                    matrix[2, 0] + matrix[0, 2],
                    matrix[2, 1] + matrix[1, 2],
                    matrix[2, 2] - matrix[0, 0] - matrix[1, 1],
                    matrix[1, 0] - matrix[0, 1],
                ],
                [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1], matrix.trace()],
            ]
        )
        / 3.0
    )
    quaternion = eigenvectors[:, np.argmax(eigenvalues)]
    if quaternion[3] < 0:
        quaternion *= -1
    return quaternion.tolist()


def _finite_vector(value: object, length: int | None, label: str) -> list[float]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        expected = f"exactly {length}" if length is not None else "only"
        raise ValueError(f"{label} must contain {expected} finite values")
    if any(not isinstance(item, int | float) or isinstance(item, bool) or not math.isfinite(item) for item in value):
        raise ValueError(f"{label} must contain finite values")
    return [float(item) for item in value]


def _quaternion_from_rpy_degrees(value: object) -> list[float]:
    roll, pitch, yaw = (math.radians(item) / 2.0 for item in _finite_vector(value, 3, "mount rpy_deg"))
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _load_test_camera_info(exported: Path) -> tuple[dict, bytes]:
    path = exported / TEST_SCENE / "camera_info.yaml"
    content = path.read_bytes()
    value = yaml.safe_load(content)
    if not isinstance(value, dict):
        raise ValueError(f"{TEST_SCENE} CameraInfo must be a YAML mapping")
    return value, content


def create_supporting_artifacts(
    *,
    mount: Path,
    exported: Path,
    camera_serial: str,
    lidar_serial: str,
    output: Path,
) -> tuple[Path, Path]:
    """Create candidate mount and intrinsics artifacts from active captured sources."""
    mount_content = mount.read_bytes()
    mount_value = yaml.safe_load(mount_content)
    if not isinstance(mount_value, dict) or mount_value.get("schema_version") != "1.0":
        raise ValueError("MID-360 mount must use schema_version 1.0")
    if (
        mount_value.get("parent_frame") != "base_link"
        or mount_value.get("body_frame") != "body"
        or mount_value.get("lidar_frame") != "livox_frame"
    ):
        raise ValueError("MID-360 mount frame contract is invalid")
    translation = _finite_vector(mount_value.get("translation_m"), 3, "mount translation_m")
    rotation = _quaternion_from_rpy_degrees(mount_value.get("rpy_deg"))

    camera_info, camera_content = _load_test_camera_info(exported)
    frame_id = camera_info.get("frame_id")
    if frame_id != "camera_front_optical_frame":
        raise ValueError("CameraInfo frame_id must equal camera_front_optical_frame")
    for dimension in ("width", "height"):
        if (
            not isinstance(camera_info.get(dimension), int)
            or isinstance(camera_info[dimension], bool)
            or camera_info[dimension] <= 0
        ):
            raise ValueError(f"CameraInfo {dimension} must be a positive integer")
    distortion_model = camera_info.get("distortion_model")
    if distortion_model != "plumb_bob":
        raise ValueError("CameraInfo distortion_model must equal plumb_bob")
    coefficients = {
        "d": _finite_vector(camera_info.get("D"), 5, "CameraInfo D"),
        "k": _finite_vector(camera_info.get("K"), 9, "CameraInfo K"),
        "r": _finite_vector(camera_info.get("R"), 9, "CameraInfo R"),
        "p": _finite_vector(camera_info.get("P"), 12, "CameraInfo P"),
    }
    if not camera_serial or not lidar_serial:
        raise ValueError("camera and lidar serial values are required")

    base_to_mid360 = {
        "schema_version": "1.0",
        "calibration_version": f"mount-{hashlib.sha256(mount_content).hexdigest()[:16]}",
        "status": "candidate",
        "device": {"name": "MID-360", "serial": lidar_serial},
        "transform": {
            "parent_frame": "base_link",
            "child_frame": "body",
            "translation": translation,
            "rotation_xyzw": rotation,
        },
    }
    front_camera_intrinsics = {
        "schema_version": "1.0",
        "calibration_version": f"camera-info-{hashlib.sha256(camera_content).hexdigest()[:16]}",
        "status": "candidate",
        "device": {"name": "front_camera", "serial": camera_serial},
        "camera_info": {
            "frame_id": frame_id,
            "width": camera_info["width"],
            "height": camera_info["height"],
            "distortion_model": distortion_model,
            **coefficients,
        },
    }
    base_path = output / "base_to_mid360.yaml"
    intrinsics_path = output / "front_camera_intrinsics.yaml"
    _exclusive_write(base_path, yaml.safe_dump(base_to_mid360, sort_keys=False).encode())
    _exclusive_write(intrinsics_path, yaml.safe_dump(front_camera_intrinsics, sort_keys=False).encode())
    return base_path, intrinsics_path


def create_candidate_artifact(
    *,
    result: Path,
    report: Path,
    mount: Path,
    capture_manifest: Path,
    camera_serial: str,
    producer_commit: str,
    parameters_sha256: str,
    output: Path,
) -> dict:
    """Compose body-to-camera with base-to-body and write a candidate artifact."""
    result_value = yaml.safe_load(result.read_bytes())
    report_value = json.loads(report.read_bytes())
    mount_value = yaml.safe_load(mount.read_bytes())
    if result_value.get("source_frame") != "body" or result_value.get("target_frame") != "camera_front_optical_frame":
        raise ValueError("solver result frame contract is invalid")
    if report_value.get("status") != "candidate" or report_value.get("result_sha256") != _sha256(result):
        raise ValueError("solver report does not bind the result")
    mount_transform = mount_value.get("transform", {})
    if mount_transform.get("parent_frame") != "base_link" or mount_transform.get("child_frame") != "body":
        raise ValueError("mount frame contract is invalid")
    if not camera_serial or len(producer_commit) != 40 or len(parameters_sha256) != 64:
        raise ValueError("camera serial and producer hashes are required")
    base_from_body = _matrix_from_quaternion(mount_transform.get("rotation_xyzw"))
    camera_from_body = np.asarray(result_value["rotation_matrix"], dtype=float)
    body_from_camera = camera_from_body.T
    base_from_camera = base_from_body @ body_from_camera
    translation = np.asarray(mount_transform.get("translation"), dtype=float) - (
        base_from_camera @ np.asarray(result_value["translation"], dtype=float)
    )
    identity_seed = hashlib.sha256(result.read_bytes() + report.read_bytes() + mount.read_bytes()).hexdigest()[:16]
    artifact = {
        "schema_version": "1.0",
        "calibration_version": f"fast-calib-{identity_seed}",
        "status": "candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": {"name": "front_camera", "serial": camera_serial},
        "transform": {
            "parent_frame": "base_link",
            "child_frame": "camera_front_optical_frame",
            "translation": translation.tolist(),
            "rotation_xyzw": _quaternion_from_matrix(base_from_camera),
        },
        "metrics": {
            "training_rmse_m": report_value["training_joint_rmse_m"],
            "test_rmse_m": report_value["test_rmse_m"],
            "correspondence_margin_m": report_value["correspondence_margin_m"],
        },
        "provenance": {
            "solver_repository": SOLVER_REPOSITORY,
            "solver_commit": SOLVER_COMMIT,
            "patch_diff_sha256": PATCH_DIFF_SHA256,
            "parameters_sha256": parameters_sha256,
            "producer_commit": producer_commit,
            "result_sha256": _sha256(result),
            "report_sha256": _sha256(report),
            "mount_sha256": _sha256(mount),
            "capture_manifest_sha256": _sha256(capture_manifest),
        },
    }
    _exclusive_write(output, yaml.safe_dump(artifact, sort_keys=False).encode())
    return artifact
