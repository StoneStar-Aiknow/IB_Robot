#!/usr/bin/env python3
"""ROS-native eye-in-hand calibration helper.

The tool samples robot poses from TF and target poses from camera images, then
solves the fixed end-effector to camera transform with OpenCV.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from dataset_tools.opencv_utils import load_opencv

CV2 = load_opencv()


@dataclass(frozen=True)
class HandEyeSample:
    """A synchronized robot and target observation."""

    base_to_ee: np.ndarray
    target_to_camera: np.ndarray
    reprojection_error_px: float | None
    detected_corners: int


@dataclass(frozen=True)
class CalibrationResult:
    """Solved hand-eye transform and validation metrics."""

    ee_to_camera_optical: np.ndarray
    ee_to_camera_link: np.ndarray
    target_translation_std_m: np.ndarray
    target_rotation_rms_deg: float
    reprojection_mean_px: float | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _method_name(value: str) -> str:
    parsed = value.lower()
    valid = {"tsai", "park", "horaud", "andreff", "daniilidis"}
    if parsed not in valid:
        raise argparse.ArgumentTypeError(f"must be one of: {', '.join(sorted(valid))}")
    return parsed


def _dictionary_name(value: str) -> str:
    parsed = value.upper()
    if not parsed.startswith("DICT_"):
        parsed = f"DICT_{parsed}"
    return parsed


def _matrix_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return matrix


def _transform_to_matrix(transform_msg) -> np.ndarray:
    translation = transform_msg.transform.translation
    rotation = transform_msg.transform.rotation
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Rotation.from_quat(
        [rotation.x, rotation.y, rotation.z, rotation.w],
    ).as_matrix()
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def _matrix_to_xyz_rpy(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(matrix[:3, 3], dtype=float)
    rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=False)
    return xyz, rpy


def _camera_link_to_optical_matrix() -> np.ndarray:
    """Return the ROS camera_link -> camera_optical_frame convention."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Rotation.from_quat([-0.5, 0.5, -0.5, 0.5]).as_matrix()
    return matrix


def _format_float(value: float) -> str:
    if abs(value) < 5e-10:
        value = 0.0
    return f"{value:.9g}"


def _format_yaml_snippet(parent_frame: str, transform: np.ndarray) -> str:
    xyz, rpy = _matrix_to_xyz_rpy(transform)
    lines = [
        "transform:",
        f"  parent_frame: {parent_frame}",
        f"  x: {_format_float(float(xyz[0]))}",
        f"  y: {_format_float(float(xyz[1]))}",
        f"  z: {_format_float(float(xyz[2]))}",
        f"  roll: {_format_float(float(rpy[0]))}",
        f"  pitch: {_format_float(float(rpy[1]))}",
        f"  yaw: {_format_float(float(rpy[2]))}",
    ]
    return "\n".join(lines)


def _rotation_rms_deg(rotations: Iterable[np.ndarray]) -> float:
    rotation_list = [Rotation.from_matrix(rot) for rot in rotations]
    if len(rotation_list) <= 1:
        return 0.0
    reference = rotation_list[0]
    errors = [(reference.inv() * item).magnitude() for item in rotation_list[1:]]
    return math.degrees(float(np.sqrt(np.mean(np.square(errors)))))


def _load_aruco_dictionary(name: str):
    if CV2 is None or not hasattr(CV2, "aruco"):
        raise RuntimeError("OpenCV with cv2.aruco support is required.")

    aruco = CV2.aruco
    dict_id = getattr(aruco, name, None)
    if dict_id is None:
        raise RuntimeError(f"Unknown ArUco dictionary: {name}")
    return aruco.getPredefinedDictionary(dict_id)


def _create_charuco_board(squares_x: int, squares_y: int, square_length: float, marker_length: float, dictionary):
    aruco = CV2.aruco
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)
    raise RuntimeError("The installed OpenCV build does not provide CharucoBoard.")


def _detect_markers(image: np.ndarray, dictionary):
    aruco = CV2.aruco
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        corners, ids, rejected = aruco.detectMarkers(image, dictionary, parameters=parameters)
    return corners, ids, rejected


def _detect_charuco_corners(image: np.ndarray, board, dictionary) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    aruco = CV2.aruco
    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, marker_ids = detector.detectBoard(image)
        marker_count = 0 if marker_ids is None else len(marker_ids)
        return charuco_corners, charuco_ids, marker_count

    corners, marker_ids, _ = _detect_markers(image, dictionary)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, 0

    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        corners,
        marker_ids,
        image,
        board,
    )
    return charuco_corners, charuco_ids, len(marker_ids)


def _estimate_charuco_pose(
    image: np.ndarray,
    board,
    dictionary,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_corners: int,
) -> tuple[np.ndarray, float | None, int]:
    charuco_corners, charuco_ids, marker_count = _detect_charuco_corners(image, board, dictionary)
    if marker_count == 0:
        raise RuntimeError("no ArUco markers detected")

    detected = 0 if charuco_ids is None else len(charuco_ids)
    if charuco_ids is None or charuco_corners is None or detected < min_corners:
        raise RuntimeError(f"only {detected} ChArUco corners detected, need at least {min_corners}")

    object_points = _charuco_object_points(board, charuco_ids)
    image_points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)

    valid, rvec, tvec = CV2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=CV2.SOLVEPNP_ITERATIVE,
    )
    if not valid:
        raise RuntimeError("ChArUco solvePnP failed")

    rotation, _ = CV2.Rodrigues(rvec)
    transform = _matrix_from_rt(rotation, tvec)
    reprojection_error = _charuco_reprojection_error(
        object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    return transform, reprojection_error, detected


def _charuco_object_points(board, charuco_ids) -> np.ndarray:
    get_corners = getattr(board, "getChessboardCorners", None)
    if get_corners is not None:
        object_points_all = np.asarray(get_corners(), dtype=np.float32)
    elif hasattr(board, "chessboardCorners"):
        object_points_all = np.asarray(board.chessboardCorners, dtype=np.float32)
    else:
        raise RuntimeError("ChArUco board does not expose chessboard corner coordinates")

    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    if len(ids) == 0:
        raise RuntimeError("no ChArUco corner ids available")
    return object_points_all[ids]


def _charuco_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs,
) -> float | None:
    projected, _ = CV2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    if len(image_points) != len(projected):
        return None
    return float(np.mean(np.linalg.norm(image_points - projected, axis=1)))


class HandEyeCaptureNode(Node):
    """Subscribe to camera data and provide synchronized capture snapshots."""

    def __init__(self, image_topic: str, camera_info_topic: str, base_frame: str, ee_frame: str, tf_timeout_sec: float):
        super().__init__("handeye_calibrator")
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_image_msg: Image | None = None
        self._latest_image: np.ndarray | None = None
        self._latest_camera_info: CameraInfo | None = None
        self._base_frame = base_frame
        self._ee_frame = ee_frame
        self._tf_timeout_sec = tf_timeout_sec
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Image, image_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_cb, qos_profile_sensor_data)

    def _image_cb(self, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to decode image: {exc}")
            return
        with self._lock:
            self._latest_image_msg = msg
            self._latest_image = image

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_camera_info = msg

    def snapshot(self) -> tuple[np.ndarray, CameraInfo, np.ndarray]:
        with self._lock:
            image_msg = self._latest_image_msg
            image = None if self._latest_image is None else self._latest_image.copy()
            camera_info = self._latest_camera_info

        if image_msg is None or image is None:
            raise RuntimeError("no image received yet")
        if camera_info is None:
            raise RuntimeError("no CameraInfo received yet")

        stamp = rclpy.time.Time.from_msg(image_msg.header.stamp)
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._ee_frame,
                stamp,
                timeout=Duration(seconds=self._tf_timeout_sec),
            )
        except TransformException as exc:
            raise RuntimeError(f"TF lookup {self._base_frame} -> {self._ee_frame} failed: {exc}") from exc

        return image, camera_info, _transform_to_matrix(transform)


def _camera_info_to_arrays(camera_info: CameraInfo) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.asarray(camera_info.k, dtype=float).reshape(3, 3)
    dist_coeffs = np.asarray(camera_info.d, dtype=float).reshape(-1, 1)
    if camera_matrix[0, 0] == 0.0 or camera_matrix[1, 1] == 0.0:
        raise RuntimeError("CameraInfo has invalid intrinsic matrix")
    return camera_matrix, dist_coeffs


def solve_handeye(samples: list[HandEyeSample], method_name: str) -> np.ndarray:
    if len(samples) < 3:
        raise RuntimeError("at least 3 samples are required")

    method_map = {
        "tsai": CV2.CALIB_HAND_EYE_TSAI,
        "park": CV2.CALIB_HAND_EYE_PARK,
        "horaud": CV2.CALIB_HAND_EYE_HORAUD,
        "andreff": CV2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": CV2.CALIB_HAND_EYE_DANIILIDIS,
    }
    rotations_gripper_to_base = [sample.base_to_ee[:3, :3] for sample in samples]
    translations_gripper_to_base = [sample.base_to_ee[:3, 3] for sample in samples]
    rotations_target_to_camera = [sample.target_to_camera[:3, :3] for sample in samples]
    translations_target_to_camera = [sample.target_to_camera[:3, 3] for sample in samples]

    rotation, translation = CV2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=method_map[method_name],
    )
    return _matrix_from_rt(rotation, translation)


def validate_result(
    samples: list[HandEyeSample], ee_to_camera_optical: np.ndarray
) -> tuple[np.ndarray, float, float | None]:
    base_to_targets = [sample.base_to_ee @ ee_to_camera_optical @ sample.target_to_camera for sample in samples]
    translations = np.asarray([item[:3, 3] for item in base_to_targets], dtype=float)
    translation_std = np.std(translations, axis=0)
    rotation_rms = _rotation_rms_deg(item[:3, :3] for item in base_to_targets)

    reprojection_values = [
        sample.reprojection_error_px for sample in samples if sample.reprojection_error_px is not None
    ]
    reprojection_mean = float(np.mean(reprojection_values)) if reprojection_values else None
    return translation_std, rotation_rms, reprojection_mean


def run_interactive(args: argparse.Namespace) -> CalibrationResult:
    if CV2 is None:
        raise RuntimeError("OpenCV is not available.")

    dictionary = _load_aruco_dictionary(args.dictionary)
    board = _create_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_length,
        args.marker_length,
        dictionary,
    )
    max_charuco_corners = max(args.squares_x - 1, 0) * max(args.squares_y - 1, 0)
    if args.min_corners > max_charuco_corners:
        raise RuntimeError(
            "--min-corners exceeds the board capacity: "
            f"{args.min_corners} requested, but a {args.squares_x}x{args.squares_y} "
            f"ChArUco board has at most {max_charuco_corners} corners",
        )

    rclpy.init(args=None)
    node = HandEyeCaptureNode(
        args.image_topic,
        args.camera_info_topic,
        args.base_frame,
        args.ee_frame,
        args.tf_timeout_sec,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    samples: list[HandEyeSample] = []
    try:
        print(
            "Using ChArUco board: "
            f"dictionary={args.dictionary}, squares={args.squares_x}x{args.squares_y}, "
            f"square_length={args.square_length:.6f}m, marker_length={args.marker_length:.6f}m, "
            f"max_corners={max_charuco_corners}",
        )
        print("Move the arm so the ChArUco board is visible, then press Enter to capture.")
        print("Use varied wrist rotations and board positions. Type 'q' then Enter to solve early.")
        while len(samples) < args.samples:
            command = input(f"[{len(samples)}/{args.samples}] capture> ").strip().lower()
            if command in {"q", "quit", "done"}:
                break
            try:
                image, camera_info, base_to_ee = node.snapshot()
                camera_matrix, dist_coeffs = _camera_info_to_arrays(camera_info)
                target_to_camera, reprojection_error, detected = _estimate_charuco_pose(
                    image,
                    board,
                    dictionary,
                    camera_matrix,
                    dist_coeffs,
                    args.min_corners,
                )
            except Exception as exc:
                print(f"  rejected: {exc}")
                continue

            if reprojection_error is not None and reprojection_error > args.max_reprojection:
                print(f"  rejected: reprojection {reprojection_error:.3f}px > {args.max_reprojection:.3f}px threshold")
                continue

            samples.append(
                HandEyeSample(
                    base_to_ee=base_to_ee,
                    target_to_camera=target_to_camera,
                    reprojection_error_px=reprojection_error,
                    detected_corners=detected,
                ),
            )
            reprojection_text = "n/a" if reprojection_error is None else f"{reprojection_error:.3f}px"
            print(f"  accepted: corners={detected}, reprojection={reprojection_text}")

        if len(samples) < args.min_samples:
            raise RuntimeError(f"need at least {args.min_samples} valid samples, got {len(samples)}")

        ee_to_camera_optical = solve_handeye(samples, args.method)
        camera_link_to_optical = _camera_link_to_optical_matrix()
        ee_to_camera_link = ee_to_camera_optical @ np.linalg.inv(camera_link_to_optical)
        translation_std, rotation_rms, reprojection_mean = validate_result(samples, ee_to_camera_optical)
        return CalibrationResult(
            ee_to_camera_optical=ee_to_camera_optical,
            ee_to_camera_link=ee_to_camera_link,
            target_translation_std_m=translation_std,
            target_rotation_rms_deg=rotation_rms,
            reprojection_mean_px=reprojection_mean,
        )
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


def write_report(path: Path, args: argparse.Namespace, result: CalibrationResult) -> None:
    parent = path.expanduser().parent
    parent.mkdir(parents=True, exist_ok=True)
    xyz_link, rpy_link = _matrix_to_xyz_rpy(result.ee_to_camera_link)
    xyz_optical, rpy_optical = _matrix_to_xyz_rpy(result.ee_to_camera_optical)
    payload = {
        "base_frame": args.base_frame,
        "ee_frame": args.ee_frame,
        "image_topic": args.image_topic,
        "camera_info_topic": args.camera_info_topic,
        "method": args.method,
        "board": {
            "type": "charuco",
            "dictionary": args.dictionary,
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length_m": args.square_length,
            "marker_length_m": args.marker_length,
        },
        "ee_to_camera_link": {
            "translation": xyz_link.tolist(),
            "rpy_xyz": rpy_link.tolist(),
            "matrix": result.ee_to_camera_link.tolist(),
        },
        "ee_to_camera_optical": {
            "translation": xyz_optical.tolist(),
            "rpy_xyz": rpy_optical.tolist(),
            "matrix": result.ee_to_camera_optical.tolist(),
        },
        "validation": {
            "target_translation_std_m": result.target_translation_std_m.tolist(),
            "target_rotation_rms_deg": result.target_rotation_rms_deg,
            "reprojection_mean_px": result.reprojection_mean_px,
        },
    }
    path.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def check_quality(
    result: CalibrationResult,
    max_translation_std_m: float,
    max_rotation_rms_deg: float,
    max_reprojection_mean_px: float,
) -> bool:
    passed = True
    for axis_name, value in zip(("x", "y", "z"), result.target_translation_std_m, strict=False):
        if value > max_translation_std_m:
            print(f"  FAIL: translation std {axis_name}={value:.5f}m > {max_translation_std_m}m")
            passed = False
    if result.target_rotation_rms_deg > max_rotation_rms_deg:
        print(f"  FAIL: rotation RMS={result.target_rotation_rms_deg:.3f}deg > {max_rotation_rms_deg}deg")
        passed = False
    if result.reprojection_mean_px is not None and result.reprojection_mean_px > max_reprojection_mean_px:
        print(f"  FAIL: reprojection mean={result.reprojection_mean_px:.3f}px > {max_reprojection_mean_px}px")
        passed = False
    return passed


def write_transform_to_config(config_path: Path, camera_name: str, parent_frame: str, transform: np.ndarray) -> None:
    xyz, rpy = _matrix_to_xyz_rpy(transform)
    new_values = {
        "parent_frame": parent_frame,
        "x": _format_float(float(xyz[0])),
        "y": _format_float(float(xyz[1])),
        "z": _format_float(float(xyz[2])),
        "roll": _format_float(float(rpy[0])),
        "pitch": _format_float(float(rpy[1])),
        "yaw": _format_float(float(rpy[2])),
    }

    text = config_path.expanduser().read_text(encoding="utf-8")

    def _find_camera_block(text: str, camera_name: str) -> int | None:
        pattern = re.compile(r"^(\s*)-\s*type:\s*camera\s*\n", re.MULTILINE)
        for m in pattern.finditer(text):
            start = m.start()
            block_indent = len(m.group(1))
            name_pattern = re.compile(
                r"^\s{0," + str(block_indent + 2) + r"}name:\s*" + re.escape(camera_name) + r"\s*\n", re.MULTILINE
            )
            if name_pattern.search(text, start):
                return start
        return None

    block_pos = _find_camera_block(text, camera_name)
    if block_pos is None:
        raise RuntimeError(f"camera '{camera_name}' not found in {config_path}")

    next_block_match = re.search(r"\n(\s*)-\s*type:", text[block_pos + 1 :])
    block_end = block_pos + 1 + next_block_match.start() if next_block_match else len(text)
    block = text[block_pos:block_end]

    transform_pattern = re.compile(
        r"(transform:\s*\n(?:[ \t]+.*\n)*)",
        re.MULTILINE,
    )
    block_match = transform_pattern.search(block)
    if block_match is None:
        raise RuntimeError(f"'transform:' not found in camera '{camera_name}' block")

    indent_match = re.search(r"([ \t]*)x:", block_match.group(1))
    indent = indent_match.group(1) if indent_match else "        "

    new_transform = (
        "transform:\n"
        f"{indent}parent_frame: {new_values['parent_frame']}\n"
        f"{indent}x: {new_values['x']}\n"
        f"{indent}y: {new_values['y']}\n"
        f"{indent}z: {new_values['z']}\n"
        f"{indent}roll: {new_values['roll']}\n"
        f"{indent}pitch: {new_values['pitch']}\n"
        f"{indent}yaw: {new_values['yaw']}\n"
    )

    new_block = block[: block_match.start()] + new_transform + block[block_match.end() :]
    text = text[:block_pos] + new_block + text[block_end:]

    config_path.expanduser().write_text(text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive ROS/OpenCV hand-in-eye calibration for IB_Robot cameras.",
    )
    parser.add_argument("--image-topic", required=True, help="Camera image topic used to observe the ChArUco board.")
    parser.add_argument("--camera-info-topic", required=True, help="CameraInfo topic matching --image-topic.")
    parser.add_argument("--base-frame", default="base", help="Robot base frame. Default: base.")
    parser.add_argument(
        "--ee-frame", default="gripper", help="End-effector frame mounted to the camera. Default: gripper."
    )
    parser.add_argument(
        "--samples", type=_positive_int, default=25, help="Target number of valid samples. Default: 25."
    )
    parser.add_argument(
        "--min-samples", type=_positive_int, default=8, help="Minimum valid samples before solving. Default: 8."
    )
    parser.add_argument("--tf-timeout-sec", type=_positive_float, default=0.5, help="TF lookup timeout. Default: 0.5.")
    parser.add_argument("--method", type=_method_name, default="park", help="OpenCV hand-eye method. Default: park.")
    parser.add_argument(
        "--dictionary", type=_dictionary_name, default="DICT_5X5_100", help="ArUco dictionary. Default: DICT_5X5_100."
    )
    parser.add_argument(
        "--squares-x", type=_positive_int, required=True, help="Number of ChArUco squares along board X."
    )
    parser.add_argument(
        "--squares-y", type=_positive_int, required=True, help="Number of ChArUco squares along board Y."
    )
    parser.add_argument(
        "--square-length", type=_positive_float, required=True, help="ChArUco square side length in meters."
    )
    parser.add_argument(
        "--marker-length", type=_positive_float, required=True, help="ChArUco marker side length in meters."
    )
    parser.add_argument(
        "--min-corners", type=_positive_int, default=8, help="Minimum detected ChArUco corners per sample. Default: 8."
    )
    parser.add_argument(
        "--max-reprojection",
        type=float,
        default=1.0,
        help="Max per-sample reprojection error in pixels. Samples above this are rejected. Default: 1.0.",
    )
    parser.add_argument("--output-json", default="", help="Optional calibration report JSON path.")
    parser.add_argument(
        "--robot-config", default="", help="Robot config YAML to auto-write the calibrated transform into."
    )
    parser.add_argument(
        "--camera-name",
        default="",
        help="Camera peripheral name in --robot-config (e.g. wrist). Required if --robot-config is set.",
    )
    parser.add_argument(
        "--max-translation-std",
        type=float,
        default=0.01,
        help="Max translation std per axis (m) to accept calibration. Default: 0.01.",
    )
    parser.add_argument(
        "--max-rotation-rms",
        type=float,
        default=2.0,
        help="Max rotation RMS (deg) to accept calibration. Default: 2.0.",
    )
    parser.add_argument(
        "--max-reprojection-mean",
        type=float,
        default=1.0,
        help="Max mean reprojection error (px) to accept calibration. Default: 1.0.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.robot_config and not args.camera_name:
        build_arg_parser().error("--camera-name is required when --robot-config is specified")

    result = run_interactive(args)

    print("\n=== Hand-eye calibration result ===")
    print("Result for robot_config peripherals[].transform (parent -> camera link):")
    print(_format_yaml_snippet(args.ee_frame, result.ee_to_camera_link))

    print("\nOptical-frame result (parent -> camera optical frame, for diagnostics only):")
    print(_format_yaml_snippet(args.ee_frame, result.ee_to_camera_optical))

    reprojection = "n/a" if result.reprojection_mean_px is None else f"{result.reprojection_mean_px:.3f}px"
    print("\nValidation:")
    print(
        "  target translation std [m]: "
        f"x={result.target_translation_std_m[0]:.5f}, "
        f"y={result.target_translation_std_m[1]:.5f}, "
        f"z={result.target_translation_std_m[2]:.5f}",
    )
    print(f"  target rotation RMS [deg]: {result.target_rotation_rms_deg:.3f}")
    print(f"  mean reprojection error: {reprojection}")

    if args.output_json:
        output_path = Path(args.output_json)
        write_report(output_path, args, result)
        print(f"\nWrote report: {output_path.expanduser()}")

    if args.robot_config:
        print("\nQuality check:")
        quality_ok = check_quality(
            result,
            args.max_translation_std,
            args.max_rotation_rms,
            args.max_reprojection_mean,
        )
        if quality_ok:
            config_path = Path(args.robot_config)
            write_transform_to_config(config_path, args.camera_name, args.ee_frame, result.ee_to_camera_link)
            print(f"  PASS: auto-wrote transform to {config_path.expanduser()}")
        else:
            print("  Quality check FAILED — config file was NOT updated.")
            print("  Adjust thresholds or re-calibrate with better samples.")


if __name__ == "__main__":
    main()
