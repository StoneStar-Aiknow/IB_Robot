"""Camera alignment helper based on ArUco markers."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_tools.opencv_utils import require_opencv_gui

# Optional ROS imports — only loaded when a ROS topic is used as input.
# True machine paths (cv2.VideoCapture) never trigger these imports.
try:
    import rclpy
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image as RosImage

    _rclpy_available = True
except ImportError:
    _rclpy_available = False

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

cv2 = None

# Override YAML base directory (never committed to git).
_OVERRIDE_BASE = Path.home() / ".ros" / "ibrobot" / "sim_camera_overrides"


@dataclass(frozen=True)
class CaptureSettings:
    """Requested OpenCV capture settings."""

    width: int | None = None
    height: int | None = None
    fps: float | None = None
    capture_format: str | None = None


@dataclass(frozen=True)
class CaptureStatus:
    """Effective OpenCV capture settings observed after the first frame."""

    width: int
    height: int
    fps: float | None
    capture_format: str | None


@dataclass(frozen=True)
class SimCalibrationTarget:
    """Resolved simulation camera contract from robot YAML."""

    camera_name: str
    camera_topic: str
    proxy_topic: str
    robot_config: str
    world_name: str
    platform: str
    width: int
    height: int
    fps: float
    calibration_domain: int


def get_status_color(error_value: float | None) -> tuple[int, int, int]:
    """Map alignment error to a UI color."""
    if error_value is None:
        return YELLOW
    if error_value < 3.0:
        return GREEN
    return RED


def compute_alignment_error(
    reference_data: dict[int, np.ndarray] | None,
    detected_markers: dict[int, np.ndarray],
) -> tuple[float | None, str]:
    """Compute average marker corner error against saved reference data."""
    if reference_data is None:
        return None, "No Reference (Press 's')"
    if not detected_markers:
        return None, "All Markers Lost"

    errors: list[float] = []
    matched_ids: list[int] = []

    for marker_id, reference_corners in reference_data.items():
        detected_corners = detected_markers.get(marker_id)
        if detected_corners is None:
            continue
        error = np.mean(np.linalg.norm(detected_corners - reference_corners, axis=1))
        errors.append(float(error))
        matched_ids.append(marker_id)

    if not errors:
        return None, f"Target IDs {sorted(reference_data.keys())} not found"

    average_error = float(np.mean(errors))
    return average_error, f"Error: {average_error:.2f}px (IDs:{matched_ids})"


def _require_opencv():
    global cv2
    if cv2 is None:
        cv2 = require_opencv_gui()
    return cv2


def _safe_destroy_window(window_name: str) -> None:
    if cv2 is None:
        return

    with contextlib.suppress(Exception):  # pragma: no cover - cleanup should not hide real failure
        cv2.destroyWindow(window_name)


def _safe_destroy_all_windows() -> None:
    if cv2 is None:
        return

    with contextlib.suppress(Exception):  # pragma: no cover - cleanup should not hide real failure
        cv2.destroyAllWindows()


def normalize_camera_source(camera_source: str) -> str | int:
    """Normalize the original camera source CLI option."""
    return int(camera_source) if camera_source.isdigit() else camera_source


def positive_int(value: str) -> int:
    """argparse type for positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    """argparse type for positive floating point values."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def capture_format_text(value: str) -> str:
    """argparse type for four-character OpenCV capture format strings."""
    if len(value) != 4 or not value.isascii():
        raise argparse.ArgumentTypeError("must be exactly four ASCII characters")
    return value.upper()


def frame_size(frame) -> tuple[int, int]:
    """Return frame size as width, height."""
    height, width = frame.shape[:2]
    return int(width), int(height)


def decode_fourcc(value: float | int | None) -> str | None:
    """Decode an OpenCV CAP_PROP_FOURCC value to a readable string."""
    try:
        code = int(value) if value is not None else 0
    except (TypeError, ValueError, OverflowError):
        return None
    if code <= 0:
        return None

    decoded = []
    for index in range(4):
        byte = (code >> (8 * index)) & 0xFF
        if 32 <= byte <= 126:
            decoded.append(chr(byte))
        else:
            decoded.append(f"\\x{byte:02x}")
    return "".join(decoded)


def format_fps(fps: float | None) -> str:
    """Format FPS without noisy trailing decimals."""
    if fps is None:
        return "unknown"
    if float(fps).is_integer():
        return str(int(fps))
    return f"{fps:.2f}".rstrip("0").rstrip(".")


def format_requested_capture(settings: CaptureSettings) -> str:
    """Describe requested capture settings for logs."""
    resolution = "default"
    if settings.width is not None or settings.height is not None:
        width = settings.width if settings.width is not None else "?"
        height = settings.height if settings.height is not None else "?"
        resolution = f"{width}x{height}"

    fps = f"@{format_fps(settings.fps)}" if settings.fps is not None else ""
    capture_format = f" {settings.capture_format}" if settings.capture_format else ""
    return f"{resolution}{fps}{capture_format}"


def format_capture_status(status: CaptureStatus) -> str:
    """Describe effective capture settings for logs."""
    capture_format = status.capture_format or "unknown"
    return f"{status.width}x{status.height}@{format_fps(status.fps)} {capture_format}"


def capture_setting_warnings(
    requested: CaptureSettings,
    actual: CaptureStatus,
) -> list[str]:
    """Return warnings for requested capture settings that did not take effect."""
    warnings: list[str] = []
    if requested.width is not None and requested.width != actual.width:
        warnings.append(f"width requested {requested.width}, actual {actual.width}")
    if requested.height is not None and requested.height != actual.height:
        warnings.append(f"height requested {requested.height}, actual {actual.height}")
    if requested.fps is not None:
        if actual.fps is None:
            warnings.append(f"fps requested {format_fps(requested.fps)}, actual unknown")
        elif abs(requested.fps - actual.fps) > 0.5:
            warnings.append(f"fps requested {format_fps(requested.fps)}, actual {format_fps(actual.fps)}")
    if requested.capture_format is not None and requested.capture_format != actual.capture_format:
        warnings.append(f"format requested {requested.capture_format}, actual {actual.capture_format or 'unknown'}")
    return warnings


def serialize_reference_payload(
    frame,
    detected_markers: dict[int, np.ndarray],
) -> dict[str, object]:
    """Serialize reference markers with the frame size used for pixel coordinates."""
    width, height = frame_size(frame)
    return {
        "image_width": width,
        "image_height": height,
        "markers": {marker_id: corners.tolist() for marker_id, corners in detected_markers.items()},
    }


def parse_reference_payload(data: object) -> tuple[dict[int, np.ndarray], tuple[int, int] | None]:
    """Parse new and legacy reference JSON payloads."""
    if not isinstance(data, dict):
        raise ValueError("reference JSON must be an object")

    image_size = None
    markers = data
    if "markers" in data:
        markers = data["markers"]
        if not isinstance(markers, dict):
            raise ValueError("reference JSON field 'markers' must be an object")
        width = data.get("image_width")
        height = data.get("image_height")
        if width is not None and height is not None:
            image_size = (int(width), int(height))

    reference_data: dict[int, np.ndarray] = {}
    for marker_id, corners in markers.items():
        reference_data[int(marker_id)] = np.array(corners, dtype=np.float32)

    return reference_data, image_size


def reference_size_status(
    reference_size: tuple[int, int] | None,
    frame,
) -> str | None:
    """Return a compact status if frame size differs from the reference size."""
    if reference_size is None:
        return None

    current_width, current_height = frame_size(frame)
    reference_width, reference_height = reference_size
    if (current_width, current_height) == (reference_width, reference_height):
        return None

    return f"Size mismatch ref {reference_width}x{reference_height} current {current_width}x{current_height}"


def reference_size_warning(
    reference_size: tuple[int, int] | None,
    frame,
) -> str | None:
    """Return a warning when pixel-coordinate alignment errors are unreliable."""
    status = reference_size_status(reference_size, frame)
    if status is None:
        return None
    return format_reference_size_warning(status)


def format_reference_size_warning(size_status: str) -> str:
    """Expand a compact size status into a user-facing warning."""
    return f"{size_status}; alignment error is unreliable. Press 's' to save a new reference."


class OpenCVFrameSource:
    """Frame source backed by cv2.VideoCapture."""

    def __init__(
        self,
        camera_source: str | int,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        capture_format: str | None = None,
    ):
        opencv = _require_opencv()
        self.camera_source = camera_source
        self.requested = CaptureSettings(
            width=width,
            height=height,
            fps=fps,
            capture_format=capture_format,
        )
        self._reported_capture = False
        self.capture = opencv.VideoCapture(camera_source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"无法打开摄像头 {camera_source}")

        self._apply_requested_settings(opencv)

    def _apply_requested_settings(self, opencv) -> None:
        if self.requested.width is not None:
            self.capture.set(opencv.CAP_PROP_FRAME_WIDTH, self.requested.width)
        if self.requested.height is not None:
            self.capture.set(opencv.CAP_PROP_FRAME_HEIGHT, self.requested.height)
        if self.requested.fps is not None:
            self.capture.set(opencv.CAP_PROP_FPS, self.requested.fps)
        if self.requested.capture_format is not None:
            fourcc_value = opencv.VideoWriter_fourcc(*self.requested.capture_format)
            self.capture.set(opencv.CAP_PROP_FOURCC, fourcc_value)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.capture.read()
        if ok and frame is not None and not self._reported_capture:
            self._reported_capture = True
            self._report_effective_capture(frame)
        return ok, frame

    def _report_effective_capture(self, frame) -> None:
        opencv = _require_opencv()
        width, height = frame_size(frame)
        fps = self.capture.get(opencv.CAP_PROP_FPS)
        actual = CaptureStatus(
            width=width,
            height=height,
            fps=fps if fps is not None else None,
            capture_format=decode_fourcc(self.capture.get(opencv.CAP_PROP_FOURCC)),
        )
        print(
            f"camera_alignment opened {self.camera_source}: "
            f"requested {format_requested_capture(self.requested)}, "
            f"actual {format_capture_status(actual)}"
        )
        warnings = capture_setting_warnings(self.requested, actual)
        if warnings:
            print(f"⚠️ camera_alignment capture request mismatch: {'; '.join(warnings)}")

    def release(self) -> None:
        self.capture.release()


# ---------------------------------------------------------------------------
# ROS topic frame source (sim path only)
# ---------------------------------------------------------------------------


def _is_ros_topic(camera_source: str) -> bool:
    """Return True when camera_source looks like a ROS topic, not a device/file.

    Rules (in order):
      - Must start with '/' (integers and relative paths → False immediately)
      - /dev/* → always a device node → False
      - Path exists on disk → local file or device → False
      - Everything else starting with '/' → treated as ROS topic
    """
    if not camera_source.startswith("/"):
        return False
    if camera_source.startswith("/dev/"):
        return False
    return not Path(camera_source).exists()


def _decode_ros_image(msg: RosImage) -> np.ndarray | None:
    """Convert sensor_msgs/Image to a BGR numpy array for OpenCV."""
    opencv = _require_opencv()
    enc = msg.encoding
    data = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        if enc == "bgr8":
            return data.reshape(msg.height, msg.width, 3)
        if enc == "rgb8":
            frame = data.reshape(msg.height, msg.width, 3)
            return opencv.cvtColor(frame, opencv.COLOR_RGB2BGR)
        if enc == "mono8":
            gray = data.reshape(msg.height, msg.width)
            return opencv.cvtColor(gray, opencv.COLOR_GRAY2BGR)
        if enc == "rgba8":
            frame = data.reshape(msg.height, msg.width, 4)
            return opencv.cvtColor(frame, opencv.COLOR_RGBA2BGR)
        if enc == "bgra8":
            frame = data.reshape(msg.height, msg.width, 4)
            return opencv.cvtColor(frame, opencv.COLOR_BGRA2BGR)
        # Unknown encoding: attempt reshape, fail gracefully
        channels = len(data) // (msg.height * msg.width)
        return data.reshape(msg.height, msg.width, channels)
    except (ValueError, Exception):
        print(f"[RosFrameSource] Cannot decode encoding={enc}")
        return None


class RosFrameSource:
    """Frame source backed by a ROS2 sensor_msgs/Image subscription.

    Interface is identical to OpenCVFrameSource: read() / release().
    Only instantiated when _is_ros_topic() returns True; never touched
    by the real-machine (OpenCV) code path.
    """

    def __init__(self, topic: str, timeout_s: float = 5.0) -> None:
        if not _rclpy_available:
            raise RuntimeError("rclpy not available — run: source /opt/ros/humble/setup.sh")
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        self._node = rclpy.create_node("camera_alignment_ros_source")
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._got_frame = threading.Event()
        self._timeout_s = timeout_s

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._node.create_subscription(RosImage, topic, self._callback, qos)
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()
        print(f"[RosFrameSource] Subscribing to {topic}")

    def _callback(self, msg: RosImage) -> None:
        frame = _decode_ros_image(msg)
        with self._lock:
            self._frame = frame
        self._got_frame.set()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return latest frame. Blocks up to timeout_s waiting for first frame.

        Returns (True, frame)  — frame ready
                (True, None)   — no frame yet, caller should continue/retry
                (False, None)  — only on actual unrecoverable error
        """
        if not self._got_frame.wait(timeout=self._timeout_s):
            # No frame yet — non-fatal, let caller retry next iteration
            with self._lock:
                if self._frame is not None:
                    return True, self._frame.copy()
            return True, None  # keep loop alive; frame may arrive later
        with self._lock:
            if self._frame is None:
                return True, None
            return True, self._frame.copy()

    def release(self) -> None:
        # Shutdown rclpy first so context.ok() becomes False, which causes
        # rclpy.spin() to exit its loop. Then join to confirm the thread is
        # done before destroying the node. Reversing the order (destroy then
        # shutdown) leaves a window where shutdown races an active spin_once.
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
        if hasattr(self, "_spin_thread") and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            self._node.destroy_node()


# ---------------------------------------------------------------------------
# Sim helpers (ArUco spawn / despawn / photo / pose override)
# ---------------------------------------------------------------------------


def _to_proxy_topic(topic: str) -> str:
    """Map a real camera topic to its proxy camera topic.

    /camera/top/image_raw  ->  /camera_align/top/image_raw
    """
    parts = topic.strip("/").split("/")
    cam_name = parts[1] if len(parts) >= 2 else "camera"
    return f"/camera_align/{cam_name}/image_raw"


def _camera_name_from_topic(topic: str) -> str:
    parts = topic.strip("/").split("/")
    return parts[1] if len(parts) >= 2 else topic.replace("/", "_")


def _robot_config_search_paths() -> list[Path]:
    candidates: list[Path] = []
    try:
        from ament_index_python.packages import get_package_share_directory

        share_dir = Path(get_package_share_directory("robot_config"))
        candidates.append(share_dir / "config" / "robots")
    except Exception:
        pass

    repo_dir = Path(__file__).resolve().parents[2] / "robot_config" / "config" / "robots"
    candidates.append(repo_dir)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _iter_robot_config_paths() -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for directory in _robot_config_search_paths():
        for path in sorted(directory.glob("*.yaml")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return paths


def _match_sim_camera(
    robot_config: dict,
    camera_topic: str,
    camera_name: str,
) -> tuple[int, dict | None]:
    peripherals = robot_config.get("peripherals", []) or []
    camera = next(
        (
            peripheral
            for peripheral in peripherals
            if peripheral.get("type") == "camera" and peripheral.get("name") == camera_name
        ),
        None,
    )
    if camera is None:
        return 0, None

    contract = robot_config.get("contract", {}) or {}
    observations = contract.get("observations", []) or []
    score = 0
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        topic_match = observation.get("topic") == camera_topic
        peripheral_match = observation.get("peripheral") == camera_name
        if topic_match and peripheral_match:
            return 3, camera
        if topic_match:
            score = max(score, 2)
        elif peripheral_match:
            score = max(score, 1)

    if camera_topic == f"/camera/{camera_name}/image_raw":
        score = max(score, 1)
    return score, camera


def _calibration_domain_for_platform(platform: str, base_domain: int) -> int:
    if platform != "mujoco":
        return base_domain

    override = os.environ.get("IBROBOT_CALIB_DOMAIN")
    domain = int(override) if override else (base_domain + 1) % 128
    if domain == base_domain:
        domain = (base_domain + 2) % 128
    return domain


def _discover_sim_calibration_target(camera_topic: str) -> SimCalibrationTarget:
    from robot_config.loader import load_robot_config_dict

    camera_name = _camera_name_from_topic(camera_topic)
    matches: list[tuple[int, Path, dict, dict]] = []
    for config_path in _iter_robot_config_paths():
        try:
            robot_config = load_robot_config_dict(config_path)
        except Exception:
            continue
        score, camera = _match_sim_camera(robot_config, camera_topic, camera_name)
        if score <= 0 or camera is None:
            continue
        matches.append((score, config_path, robot_config, camera))

    if not matches:
        raise RuntimeError(
            f"无法从 robot_config YAML 里找到相机话题 {camera_topic}。"
            " 请先 build 并确认该 topic 已写入 contract/peripherals。"
        )

    matches.sort(key=lambda item: item[0], reverse=True)
    best_score = matches[0][0]
    best = [item for item in matches if item[0] == best_score]
    if len(best) > 1:
        names = ", ".join(sorted(item[2].get("name", item[1].stem) for item in best))
        raise RuntimeError(f"相机话题 {camera_topic} 对应多个 robot_config，无法唯一判断: {names}")

    _, config_path, robot_config, camera = best[0]
    platform = str((robot_config.get("simulation", {}) or {}).get("platform") or "gazebo").strip().lower()
    world_name = str(robot_config.get("gazebo_world_name") or "demo")
    base_domain = int(os.environ.get("ROS_DOMAIN_ID", "0") or "0")
    robot_name = str(robot_config.get("name") or config_path.stem)
    return SimCalibrationTarget(
        camera_name=camera_name,
        camera_topic=camera_topic,
        proxy_topic=_to_proxy_topic(camera_topic),
        robot_config=robot_name,
        world_name=world_name,
        platform=platform,
        width=int(camera.get("width", 640) or 640),
        height=int(camera.get("height", 480) or 480),
        fps=float(camera.get("fps", 30) or 30),
        calibration_domain=_calibration_domain_for_platform(platform, base_domain),
    )


class ManagedSimCalibrationHelper:
    """Parent-managed sim helper subprocess for the unified sim workflow."""

    def __init__(self, target: SimCalibrationTarget, base_env: dict[str, str]) -> None:
        self._target = target
        self._base_env = base_env.copy()
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        env = self._base_env.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if self._target.platform == "mujoco":
            env["IBROBOT_CALIB_DOMAIN"] = str(self._target.calibration_domain)

        cmd = [
            "ros2",
            "run",
            "sim_models",
            "sim_camera_adjuster",
            "--camera",
            self._target.camera_topic,
            "--robot-config",
            self._target.robot_config,
            "--world",
            self._target.world_name,
        ]
        self._proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)

    def ensure_running(self) -> None:
        if self._proc is None:
            raise RuntimeError("sim calibration helper 尚未启动")
        code = self._proc.poll()
        if code is not None:
            raise RuntimeError(f"sim calibration helper exited early with code {code}")

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, signal.SIGINT)
            try:
                self._proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._proc.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._proc.wait(timeout=3.0)
        self._proc = None


def _wait_for_sim_frame(
    capture: RosFrameSource,
    helper: ManagedSimCalibrationHelper,
    topic: str,
    timeout_s: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        helper.ensure_running()
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"读取仿真画面失败: {topic}")
        if frame is not None:
            return
    helper.ensure_running()
    raise TimeoutError(f"timed out waiting for proxy camera frames on {topic}")


def _print_sim_capture_settings_ignored(parsed: argparse.Namespace) -> None:
    requested = []
    if parsed.width is not None:
        requested.append(f"width={parsed.width}")
    if parsed.height is not None:
        requested.append(f"height={parsed.height}")
    if parsed.fps is not None:
        requested.append(f"fps={format_fps(parsed.fps)}")
    if parsed.capture_format is not None:
        requested.append(f"format={parsed.capture_format}")
    if requested:
        print("[sim] capture settings ignored in sim mode; using YAML camera geometry: " + ", ".join(requested))


def _try_spawn_aruco(world_name: str = "demo") -> str | None:
    """Attempt to spawn ArUco A4 in Gazebo; return model_name or None on failure."""
    try:
        from sim_models.aruco_spawner import spawn_aruco_gazebo

        model_name = spawn_aruco_gazebo(world_name=world_name)
        print(f"[sim] ArUco A4 spawned: {model_name}")
        return model_name
    except ImportError:
        print("[sim] sim_models not found — ArUco tag won't be spawned")
    except Exception as exc:
        print(f"[sim] ArUco spawn failed (non-fatal): {exc}")
    return None


def _try_despawn_aruco(model_name: str, world_name: str = "demo") -> None:
    try:
        from sim_models.aruco_spawner import despawn_aruco_gazebo

        despawn_aruco_gazebo(model_name, world_name=world_name)
        print(f"[sim] ArUco A4 despawned: {model_name}")
    except Exception as exc:
        print(f"[sim] ArUco despawn failed (non-fatal): {exc}")


def _save_photo(frame: np.ndarray) -> Path:
    """Save current frame as a PNG in the current working directory; returns Path."""
    opencv = _require_opencv()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{ts}.png"
    opencv.imwrite(filename, frame)
    print(f"[camera_alignment] Photo saved: {filename}")
    return Path(filename)


def _save_pose_override(camera_source: str) -> None:
    """Write a camera pose override YAML stub for the current sim session.

    The file records the camera topic (which encodes the camera name) and a
    placeholder pose.  Phase 3 (trackbar) will overwrite pose values in-place.

    Path: ~/.ros/ibrobot/sim_camera_overrides/<camera_name>.yaml
    Only called in the ROS/sim code path — never reached from real-machine path.
    """
    # Derive a short camera name from the topic, e.g. /camera/top/image_raw → top
    parts = camera_source.strip("/").split("/")
    camera_name = parts[1] if len(parts) >= 2 else camera_source.replace("/", "_")

    override_dir = _OVERRIDE_BASE
    override_dir.mkdir(parents=True, exist_ok=True)
    override_file = override_dir / f"{camera_name}.yaml"

    content = (
        f"# Camera pose override — written by camera_alignment (Phase 2 stub)\n"
        f"# Phase 3 (sim_camera_adjuster) will overwrite this with real values.\n"
        f"# This file is intentionally skipped by sim_camera_adjuster until saved via trackbar.\n"
        f"_is_stub: true\n"
        f"camera: {camera_name}\n"
        f"topic: {camera_source}\n"
        f'saved_at: "{datetime.datetime.now().isoformat(timespec="seconds")}"\n'
        f"pose:\n"
        f"  x: 0.0\n"
        f"  y: 0.0\n"
        f"  z: 0.0\n"
        f"  roll: 0.0\n"
        f"  pitch: 0.0\n"
        f"  yaw: 0.0\n"
        f"fovy_deg: 60.0\n"
    )
    override_file.write_text(content, encoding="utf-8")
    print(f"[sim] Pose override stub saved: {override_file}")


def _save_frame_as_reference(aligner: MultiCameraAligner, frame) -> None:
    """Save current frame directly as reference image without requiring ArUco (markerless)."""
    opencv = _require_opencv()
    opencv.imwrite(str(aligner.reference_image_path), frame)
    print(f"✅ 参考图已保存: {aligner.reference_image_path}")


def _read_override_pose(camera_source: str) -> dict | None:
    """Read saved pose from override YAML. Returns None if not saved or is stub."""
    parts = camera_source.strip("/").split("/")
    cam_name = parts[1] if len(parts) >= 2 else camera_source.replace("/", "_")
    override_file = _OVERRIDE_BASE / f"{cam_name}.yaml"
    if not override_file.exists():
        return None
    try:
        import yaml as _yaml

        data = _yaml.safe_load(override_file.read_text(encoding="utf-8"))
        if not data or data.get("_is_stub"):
            return None
        return data
    except Exception:
        return None


def create_aruco_detector():
    """Create a detector that works across OpenCV ArUco API versions."""
    opencv = _require_opencv()
    if hasattr(opencv.aruco, "DetectorParameters"):
        parameters = opencv.aruco.DetectorParameters()
    else:
        parameters = opencv.aruco.DetectorParameters_create()

    if hasattr(opencv.aruco, "ArucoDetector"):
        detector = opencv.aruco.ArucoDetector(
            opencv.aruco.getPredefinedDictionary(opencv.aruco.DICT_4X4_50),
            parameters,
        )
        return detector, parameters

    return None, parameters


class MultiCameraAligner:
    """Interactive marker-based camera alignment helper."""

    def __init__(
        self,
        reference_path: str | Path = "camera_reference_multi.json",
        reference_image_path: str | Path = "reference_img.png",
    ):
        opencv = _require_opencv()
        self.reference_path = Path(reference_path)
        self.reference_image_path = Path(reference_image_path)
        self.dictionary = opencv.aruco.getPredefinedDictionary(opencv.aruco.DICT_4X4_50)
        self.detector, self.parameters = create_aruco_detector()
        self.reference_size: tuple[int, int] | None = None
        self._reference_size_warning_printed = False
        self.reference_data = self.load_reference()

    def load_reference(self) -> dict[int, np.ndarray] | None:
        if not self.reference_path.exists():
            self.reference_size = None
            return None

        with open(self.reference_path, encoding="utf-8") as file:
            data = json.load(file)

        reference_data, self.reference_size = parse_reference_payload(data)
        return reference_data

    def detect_markers(self, frame) -> tuple[dict[int, np.ndarray], np.ndarray | None, list]:
        opencv = _require_opencv()
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(frame)
        else:
            corners, ids, rejected = opencv.aruco.detectMarkers(
                frame,
                self.dictionary,
                parameters=self.parameters,
            )
        if ids is None:
            return {}, None, rejected
        marker_ids = ids.flatten()
        detected = {int(marker_ids[index]): corners[index][0] for index in range(len(marker_ids))}
        return detected, ids, rejected

    def save_reference(self, frame) -> bool:
        opencv = _require_opencv()
        detected, _, _ = self.detect_markers(frame)
        if not detected:
            print("❌ 错误：当前画面没看到任何 ArUco 码，无法保存！")
            return False

        serialized = serialize_reference_payload(frame, detected)
        with open(self.reference_path, "w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=2, ensure_ascii=False)
        opencv.imwrite(str(self.reference_image_path), frame)
        self.reference_data = self.load_reference()
        print(f"✅ 基准已更新，保存了 {len(detected)} 个 marker。")
        return True

    def get_alignment_error(self, frame) -> tuple[float | None, str]:
        detected, _, _ = self.detect_markers(frame)
        return self.get_alignment_status(detected, frame)

    def get_alignment_status(
        self,
        detected_markers: dict[int, np.ndarray],
        frame,
    ) -> tuple[float | None, str]:
        error_value, status = compute_alignment_error(self.reference_data, detected_markers)
        size_status = reference_size_status(self.reference_size, frame)
        if size_status is None:
            return error_value, status

        warning = format_reference_size_warning(size_status)
        if not self._reference_size_warning_printed:
            self._reference_size_warning_printed = True
            print(f"⚠️ {warning}")
        return error_value, f"{status} | {size_status}"

    def run_ghosting_ui(
        self,
        capture,
        on_save=None,
        markerless: bool = False,
        override_read_fn=None,
    ) -> None:
        """Show ghosting overlay.

        Args:
            on_save: callback(frame) invoked on 's' press; if None, falls back to save_reference().
            markerless: when True, skip ArUco error computation and show subjective-alignment hint.
            override_read_fn: callable() -> dict | None, called each frame to get latest saved pose
                              (for sim YAML overlay). None = no overlay.
        """
        opencv = _require_opencv()
        reference_image = opencv.imread(str(self.reference_image_path))
        if reference_image is None:
            print("❌ 找不到参考图，请先按 's' 保存或提供 --reference-image-path")
            return

        if markerless:
            print(">>> GHOST MODE  Subjective alignment (no metric)  s=update ref  q=quit")
        else:
            print(">>> GHOST MODE  s=save reference  q=quit")

        _prev_saved_state = None  # detect transition to SAVED

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            reference_resized = opencv.resize(reference_image, (frame.shape[1], frame.shape[0]))
            ghost = opencv.addWeighted(frame, 0.5, reference_resized, 0.5, 0)

            # --- Line 1: mode / error status (ASCII only for putText) ---
            if markerless:
                line1 = "GHOST | Subjective alignment (no metric)"
                line1_color = YELLOW
            else:
                error_value, status = self.get_alignment_error(frame)
                line1 = f"GHOST MODE: {status}"
                line1_color = get_status_color(error_value)

            opencv.putText(ghost, line1, (20, 40), opencv.FONT_HERSHEY_SIMPLEX, 0.65, line1_color, 2)

            # --- Line 2: YAML pose status (sim only) ---
            if override_read_fn is not None:
                saved = override_read_fn()
                if saved:
                    p = saved.get("pose", {})
                    fovy = saved.get("fovy_deg", "?")
                    line2 = (
                        f"Pose saved  x={p.get('x', 0) * 100:.1f}cm "
                        f"y={p.get('y', 0) * 100:.1f}cm  z={p.get('z', 0) * 100:.1f}cm"
                        f"  fovy={fovy}deg  |  Press q to exit"
                    )
                    line2_color = GREEN
                    if _prev_saved_state is None:  # first frame after save
                        print("[sim] Pose confirmed saved. Press q to exit ghost mode.")
                    _prev_saved_state = saved
                else:
                    line2 = "No pose saved -- drag adjuster sliders, then press S in adjuster"
                    line2_color = (0, 165, 255)  # orange
                    _prev_saved_state = None
                opencv.putText(ghost, line2, (20, 72), opencv.FONT_HERSHEY_SIMPLEX, 0.50, line2_color, 1)

            opencv.imshow("Ghosting_Mode", ghost)

            key = opencv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if on_save is not None:
                    on_save(frame)
                else:
                    self.save_reference(frame)  # real-machine marker mode

        _safe_destroy_window("Ghosting_Mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Marker-based camera alignment helper",
    )
    parser.add_argument(
        "--camera-source",
        dest="cameras_index_or_path",
        help="Camera source: OpenCV camera index/device/file path, or ROS image topic with --use_sim",
    )
    parser.add_argument(
        "--cameras_index_or_path",
        dest="cameras_index_or_path",
        help="Deprecated alias for --camera-source",
    )
    parser.add_argument(
        "--reference-path",
        default="camera_reference_multi.json",
        help="Path to the saved reference marker JSON",
    )
    parser.add_argument(
        "--reference-image-path",
        default="reference_img.png",
        help="Path to the saved reference image",
    )
    parser.add_argument(
        "--width",
        type=positive_int,
        help="Requested capture width in pixels",
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        help="Requested capture height in pixels",
    )
    parser.add_argument(
        "--fps",
        type=positive_float,
        help="Requested capture frame rate",
    )
    parser.add_argument(
        "--format",
        dest="capture_format",
        type=capture_format_text,
        help="Requested capture format, for example MJPG or YUYV",
    )
    parser.add_argument(
        "--fourcc",
        dest="capture_format",
        type=capture_format_text,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--markerless",
        action="store_true",
        help=(
            "Markerless alignment: skip ArUco detection. "
            "'s' saves the current frame as reference image directly. "
            "Works for both real-machine and sim; sim additionally shows YAML pose overlay in ghost mode."
        ),
    )
    parser.add_argument(
        "--use_sim",
        action="store_true",
        help="Treat the input as a ROS image topic for simulation alignment",
    )
    return parser


def _validate_mode_args(parser: argparse.ArgumentParser, parsed: argparse.Namespace) -> bool:
    camera_source = parsed.cameras_index_or_path
    if camera_source is None:
        parser.error("--camera-source is required")
    looks_like_ros_topic = _is_ros_topic(camera_source)
    if looks_like_ros_topic and not parsed.use_sim:
        parser.error("ROS image topics require --use_sim, for example: --use_sim --camera-source /camera/top/image_raw")
    if parsed.use_sim and not looks_like_ros_topic:
        parser.error("--use_sim requires a ROS image topic such as /camera/top/image_raw")
    return parsed.use_sim


def _create_aligner(parsed: argparse.Namespace) -> MultiCameraAligner:
    aligner = MultiCameraAligner(
        reference_path=parsed.reference_path,
        reference_image_path=parsed.reference_image_path,
    )

    # Auto-extract ArUco reference data from provided reference image (marker mode only)
    if not parsed.markerless and aligner.reference_data is None and aligner.reference_image_path.exists():
        _ref_opencv = _require_opencv()
        _ref_img = _ref_opencv.imread(str(aligner.reference_image_path))
        if _ref_img is not None:
            _detected, _, _ = aligner.detect_markers(_ref_img)
            if _detected:
                _serialized = serialize_reference_payload(_ref_img, _detected)
                with open(aligner.reference_path, "w", encoding="utf-8") as _f:
                    json.dump(_serialized, _f, indent=2, ensure_ascii=False)
                aligner.reference_data = aligner.load_reference()
                print(f"✅ 已从参考图像自动提取 {len(_detected)} 个 ArUco marker 数据")
            else:
                print("[warn] 参考图像中未检测到 ArUco marker，请先对准标记物再按 's'")
    return aligner


def _print_runtime_controls(
    markerless: bool,
    is_sim: bool,
    aligner: MultiCameraAligner,
    original_cameras_source: str,
) -> None:
    if markerless:
        print("s: 保存当前帧为参考图" + ("（仿真：会读取 adjuster 已保存参数并打印）" if is_sim else ""))
        if is_sim and aligner.reference_image_path.exists():
            print(f"[sim] 参考图已就绪: {aligner.reference_image_path}  直接按 v 进入虚影")
            print(
                "[sim] 需要先启动 adjuster: ros2 run sim_models sim_camera_adjuster --camera " + original_cameras_source
            )
    else:
        print("s: 保存当前 marker 作为基准")
    print("p: 拍照存图（markerless 无 reference 时首张照片自动设为参考）")
    if is_sim:
        print("b: 开关 A4 标定纸")
    print("v: 进入虚影对齐模式")
    print("q: 退出")


def _run_alignment_loop(
    capture,
    aligner: MultiCameraAligner,
    markerless: bool,
    is_sim: bool,
    original_cameras_source: str,
    sim_target: SimCalibrationTarget | None = None,
    model_name: str | None = None,
    paper_enabled: bool = False,
) -> str | None:
    opencv = _require_opencv()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None:
            # No frame yet (bridge/topic not ready); keep waiting
            time.sleep(0.05)
            continue

        display_frame = frame.copy()

        if markerless:
            ref_exists = aligner.reference_image_path.exists()
            status = (
                f"MARKERLESS | ref: {aligner.reference_image_path.name}"
                if ref_exists
                else "MARKERLESS | no ref -- press p to capture or provide --reference-image-path"
            )
            color = YELLOW
        else:
            detected, ids, _ = aligner.detect_markers(frame)
            error_value, status = aligner.get_alignment_status(detected, frame)
            color = get_status_color(error_value)
            if ids is not None:
                marker_corners = [corners.reshape(1, 4, 2) for corners in detected.values()]
                opencv.aruco.drawDetectedMarkers(display_frame, marker_corners, ids)

        opencv.putText(display_frame, status, (10, 30), opencv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        opencv.imshow("Calibration_Monitor", display_frame)

        key = opencv.waitKey(1) & 0xFF
        if key == ord("s"):
            if markerless:
                _save_frame_as_reference(aligner, frame)
                if is_sim:
                    saved = _read_override_pose(original_cameras_source)
                    if saved:
                        p = saved.get("pose", {})
                        print(
                            f"[sim] 当前已保存相机参数: "
                            f"x={p.get('x', 0):.3f} y={p.get('y', 0):.3f} "
                            f"z={p.get('z', 0):.3f}  fovy={saved.get('fovy_deg')}°"
                        )
                    else:
                        print("[sim] ⚠ 尚未保存仿真相机参数，请在 adjuster 窗口按 S")
            else:
                aligner.save_reference(frame)
        elif key == ord("p"):
            photo_path = _save_photo(frame)
            # In markerless mode, if no reference image exists yet, use the just-taken photo
            if markerless and not aligner.reference_image_path.exists():
                aligner.reference_image_path = photo_path
                print(f"[markerless] Reference image set to: {photo_path}  (press v for ghost)")
        elif key == ord("b") and is_sim and sim_target is not None:
            if paper_enabled and model_name is not None:
                _try_despawn_aruco(model_name, world_name=sim_target.world_name)
                model_name = None
                paper_enabled = False
            else:
                model_name = _try_spawn_aruco(world_name=sim_target.world_name)
                paper_enabled = model_name is not None
        elif key == ord("v"):
            if not aligner.reference_image_path.exists():
                print("⚠ 请先按 'p' 拍一张参考图，或通过 --reference-image-path 提供")
            else:

                def _on_ghost_save(frame):
                    if markerless:
                        _save_frame_as_reference(aligner, frame)
                    else:
                        aligner.save_reference(frame)

                aligner.run_ghosting_ui(
                    capture,
                    on_save=_on_ghost_save,
                    markerless=markerless,
                    override_read_fn=((lambda: _read_override_pose(original_cameras_source)) if is_sim else None),
                )
        elif key == ord("q"):
            break
    return model_name


def _run_sim_mode(parsed: argparse.Namespace) -> int:
    original_cameras_source = parsed.cameras_index_or_path
    original_ros_domain = os.environ.get("ROS_DOMAIN_ID")
    helper_base_env = dict(os.environ)
    sim_helper: ManagedSimCalibrationHelper | None = None
    model_name: str | None = None

    sim_target = _discover_sim_calibration_target(original_cameras_source)
    print(
        f"[sim] using {sim_target.platform} runtime "
        f"(robot_config={sim_target.robot_config}, world={sim_target.world_name})"
    )
    print(f"[sim] proxy topic: {sim_target.proxy_topic}")
    _print_sim_capture_settings_ignored(parsed)
    if sim_target.platform == "mujoco":
        os.environ["ROS_DOMAIN_ID"] = str(sim_target.calibration_domain)
        print(f"[sim] switched camera_alignment to calibration ROS_DOMAIN_ID={sim_target.calibration_domain}")

    capture = RosFrameSource(sim_target.proxy_topic)
    try:
        sim_helper = ManagedSimCalibrationHelper(sim_target, base_env=helper_base_env)
        sim_helper.start()
        _wait_for_sim_frame(capture, sim_helper, sim_target.proxy_topic)
        model_name = _try_spawn_aruco(world_name=sim_target.world_name)
        aligner = _create_aligner(parsed)
        _print_runtime_controls(parsed.markerless, True, aligner, original_cameras_source)
        model_name = _run_alignment_loop(
            capture,
            aligner,
            parsed.markerless,
            True,
            original_cameras_source,
            sim_target=sim_target,
            model_name=model_name,
            paper_enabled=model_name is not None,
        )
    finally:
        capture.release()
        if model_name is not None:
            _try_despawn_aruco(model_name, world_name=sim_target.world_name)
        if sim_helper is not None:
            sim_helper.stop()
        if original_ros_domain is not None:
            os.environ["ROS_DOMAIN_ID"] = original_ros_domain
        else:
            os.environ.pop("ROS_DOMAIN_ID", None)
        _safe_destroy_all_windows()

    return 0


def _run_real_mode(parsed: argparse.Namespace) -> int:
    cameras_source = parsed.cameras_index_or_path
    capture = OpenCVFrameSource(
        normalize_camera_source(cameras_source),
        width=parsed.width,
        height=parsed.height,
        fps=parsed.fps,
        capture_format=parsed.capture_format,
    )
    try:
        aligner = _create_aligner(parsed)
        _print_runtime_controls(parsed.markerless, False, aligner, cameras_source)
        _run_alignment_loop(capture, aligner, parsed.markerless, False, cameras_source)
    finally:
        capture.release()
        _safe_destroy_all_windows()

    return 0


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    _require_opencv()

    if _validate_mode_args(parser, parsed):
        return _run_sim_mode(parsed)
    return _run_real_mode(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
